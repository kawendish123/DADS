from utils import inference_utils
from dads_framework.dads import algorithm_DSL, get_partition_points, algorithm_DSH, algorithm_DADS
from dads_framework.graph_construct import get_layers_latency
import net.net_utils as net
import torch
import torch.nn as nn
import pickle
from net.net_utils import get_speed

class HRNet_EdgeModel(nn.Module):
    """边缘端动态执行引擎"""

    def __init__(self, model, partition_edges):
        super().__init__()
        self.model = model
        self.send_nodes = [edge[0] for edge in partition_edges]

        # 🌟 修复 1：如果 send_nodes 为空（纯边缘模式），则执行到底
        if self.send_nodes:
            self.max_node = max(self.send_nodes)
        else:
            self.max_node = float('inf')  # 没有拦截点，直接跑到天荒地老

    def forward(self, x):
        dict_layer_output = {0: x}
        edge_outputs = {}

        if 0 in self.send_nodes:
            edge_outputs[0] = x

        current_input = x

        for layer_index, layer in enumerate(self.model):
            logic_idx = layer_index + 1
            if logic_idx > self.max_node:
                break

            if self.model.has_dag_topology and logic_idx in self.model.dag_dict.keys():
                pre_input_cond = self.model.dag_dict[logic_idx]
                if isinstance(pre_input_cond, str) and pre_input_cond not in dict_layer_output:
                    fusion_deps = self.model.dag_dict[pre_input_cond]
                    fusion_inputs = [dict_layer_output[dep] for dep in fusion_deps]
                    fusion_op = self.model.merge_ops[pre_input_cond]
                    dict_layer_output[pre_input_cond] = fusion_op(fusion_inputs)

                if isinstance(pre_input_cond, list):
                    current_input = [dict_layer_output[idx] for idx in pre_input_cond]
                else:
                    current_input = dict_layer_output[pre_input_cond]

            output = layer(current_input)

            if self.model.has_dag_topology and logic_idx in self.model.record_output_list:
                dict_layer_output[logic_idx] = output

            if logic_idx in self.send_nodes:
                edge_outputs[logic_idx] = output

            # 🌟 修复 2：如果是纯边缘模式，强制将最后一层的终极输出打包发给云端交差
            if not self.send_nodes and layer_index == len(self.model) - 1:
                edge_outputs[logic_idx] = output

            current_input = output

        return edge_outputs


class HRNet_CloudModel(nn.Module):
    """云端动态执行引擎"""

    def __init__(self, model, partition_edges):
        super().__init__()
        self.model = model

    def forward(self, edge_outputs):
        dict_layer_output = edge_outputs.copy()

        current_input = dict_layer_output.get(0, None)
        final_output = current_input

        for layer_index, layer in enumerate(self.model):
            logic_idx = layer_index + 1

            if logic_idx in dict_layer_output:
                current_input = dict_layer_output[logic_idx]
                # 🌟 修复 3：如果直接命中边缘端传来的结果，必须同步更新 final_output
                final_output = current_input
                continue

            if self.model.has_dag_topology and logic_idx in self.model.dag_dict.keys():
                pre_input_cond = self.model.dag_dict[logic_idx]

                if isinstance(pre_input_cond, str) and pre_input_cond not in dict_layer_output:
                    fusion_deps = self.model.dag_dict[pre_input_cond]
                    if not all(dep in dict_layer_output for dep in fusion_deps):
                        current_input = None
                        continue
                    fusion_inputs = [dict_layer_output[dep] for dep in fusion_deps]
                    fusion_op = self.model.merge_ops[pre_input_cond]
                    dict_layer_output[pre_input_cond] = fusion_op(fusion_inputs)

                if isinstance(pre_input_cond, list):
                    if not all(dep in dict_layer_output for dep in pre_input_cond):
                        current_input = None
                        continue
                    current_input = [dict_layer_output[idx] for idx in pre_input_cond]
                else:
                    if pre_input_cond not in dict_layer_output:
                        current_input = None
                        continue
                    current_input = dict_layer_output[pre_input_cond]
            else:
                if current_input is None:
                    continue

            # 真正执行计算
            output = layer(current_input)

            if self.model.has_dag_topology and logic_idx in self.model.record_output_list:
                dict_layer_output[logic_idx] = output

            current_input = output
            final_output = output

        return final_output

def start_server(socket_server, device):
    """
    开始监听客户端传来的消息
    一般仅在 cloud_api.py 中直接调用
    :param socket_server: socket服务端
    :param device: 使用本地的cpu运行还是cuda运行
    :return: None
    """
    # 等待客户端连接
    conn, client = net.wait_client(socket_server)

    # 接收模型类型
    model_type = net.get_short_data(conn)
    print(f"get model type successfully.")

    # 读取模型
    model = inference_utils.get_dnn_model(model_type)

    # 获取云端各层时延
    cloud_latency_list = get_layers_latency(model, device=device)
    net.send_data(conn, cloud_latency_list, "model latency on the cloud device.")

    # 接收模型分层点
    model_partition_edge = net.get_short_data(conn)
    print(f"get partition point successfully.")
    conn.sendall("ok".encode())
    # 获取划分后的边缘端模型和云端模型
    # _, cloud_model = inference_utils.model_partition(model, model_partition_edge)
    # cloud_model = cloud_model.to(device)
    cloud_model = HRNet_CloudModel(model, model_partition_edge).to(device)

    # 接收中间数据并返回传输时延
    edge_output, transfer_latency = net.get_data(conn)

    # 避免连续发送两个消息 防止消息粘包
    conn.recv(40)

    print(f"get edge_output and transfer latency successfully.")
    net.send_short_data(conn, transfer_latency, "transfer latency")

    # 避免连续发送两个消息 防止消息粘包
    conn.recv(40)

    if isinstance(edge_output, dict):
        # 1. 确保字典里的所有包裹都在正确的设备上
        for k in edge_output:
            edge_output[k] = edge_output[k].to(device)

        # 2. 自定义字典预热 (跑 10 次即可)
        cloud_model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = cloud_model(edge_output)

        # 3. 自定义精确 GPU 测速 (跑 100 次求平均)
        import time
        start_event = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        end_event = torch.cuda.Event(enable_timing=True) if device == "cuda" else None

        if device == "cuda":
            start_event.record()
        else:
            start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(100):
                cloud_output = cloud_model(edge_output)

        if device == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            cloud_latency = start_event.elapsed_time(end_event) / 100.0
        else:
            end_time = time.perf_counter()
            cloud_latency = (end_time - start_time) * 1000 / 100.0

    else:
        # 如果跑的是普通单路模型 (如 VGG)，依然使用原版逻辑
        edge_output = edge_output.to(device)
        inference_utils.warmUp(cloud_model, edge_output, device)
        cloud_output, cloud_latency = inference_utils.recordTime(cloud_model, edge_output, device, epoch_cpu=30,
                                                                 epoch_gpu=100)
    # ================= 🌟 核心补丁区域 结束 =================
    # inference_utils.warmUp(cloud_model, edge_output, device)

    # 记录云端推理时延
    # cloud_output, cloud_latency = inference_utils.recordTime(cloud_model, edge_output, device, epoch_cpu=30,
    #                                                          epoch_gpu=100)
    print(f"{model_type} 在云端设备上推理完成 - {cloud_latency:.3f} ms")
    # net.send_short_data(conn, cloud_latency, "cloud latency")
    net.send_data(conn, cloud_latency, "cloud latency")

    print("================= DNN Collaborative Inference Finished. ===================")


def start_client(ip, port, input_x, model_type, upload_bandwidth, device,Q,
                 edge_only=False,cloud_only=False):
    """
    启动一个client客户端 向server端发起推理请求
    一般仅在 edge_api.py 中直接调用
    :param ip: server端的ip地址
    :param port: server端的端口地址
    :param input_x: 初始输入
    :param model_type: 选用的模型类型
    :param upload_bandwidth 上传带宽
    :param device: 在本地cpu运行还是cuda运行
    :return: None
    """
    # 读取模型
    model = inference_utils.get_dnn_model(model_type)
    # 和云端建立连接
    conn = net.get_socket_client(ip, port)
    layer_num = len(model)
    # 发送一个数据请求云端的各层推理时延
    net.send_short_data(conn, model_type, msg="model type")
    edge_latency_list = get_layers_latency(model, device=device)  # 计算出边缘端的时延参数
    cloud_latency_list,_ = net.get_data(conn)  # 接受到云端的时延参数

    if edge_only:
        print("\n>>> 强行切分：纯边缘设备计算模式 (Edge-Only)")
        # 切分点在最后一层之后，意味着模型全部留在边缘端
        model_partition_edge = []

    elif cloud_only:
        print("\n>>> 强行切分：纯云端设备计算模式 (Cloud-Only)")
        # 切分点在第0层，意味着边缘端为空，直接传输原始 Input
        model_partition_edge = [(0, 1)]

    else:
        print("\n>>> 动态切分：DADS 端云协同优化模式")
        graph_partition_edge, dict_node_layer = algorithm_DADS(
            model, input_x, edge_latency_list, cloud_latency_list,
            bandwidth=upload_bandwidth, Q=Q
        )
        model_partition_edge = get_partition_points(graph_partition_edge, dict_node_layer)
    print(f"partition edges : {model_partition_edge}")


    # 发送划分点
    net.send_short_data(conn, model_partition_edge, msg="partition strategy")
    conn.recv(40)
    # 获取划分后的边缘端模型和云端模型
    # edge_model, _ = inference_utils.model_partition(model, model_partition_edge)
    # edge_model = edge_model.to(device)
    edge_model = HRNet_EdgeModel(model, model_partition_edge).to(device)

    # 开始边缘端的推理 首先进行预热
    inference_utils.warmUp(edge_model, input_x, device)
    edge_output, edge_latency = inference_utils.recordTime(edge_model, input_x, device, epoch_cpu=30, epoch_gpu=100)
    print(f"{model_type} 在边缘端设备上推理完成 - {edge_latency:.3f} ms")

    # # 发送中间数据
    # net.send_data(conn, edge_output, "edge output")
    #
    # # 避免连续接收两个消息 防止消息粘包
    # conn.sendall("avoid  sticky".encode())
    #
    # transfer_latency = net.get_short_data(conn)
    # print(f"{model_type} 传输完成 - {transfer_latency:.3f} ms")

    edge_output_bytes = pickle.dumps(edge_output)
    data_size = len(edge_output_bytes)

    # 2. 根据你设定的带宽，获取传输速度 (Bytes/ms)
    speed_Bpms = get_speed("wifi", upload_bandwidth)

    # 3. 计算理论传输时延 (毫秒)
    simulated_transfer_latency = data_size / speed_Bpms

    # ==========================================================
    # 依然执行真实的物理发送（保持 Server 端能收到数据继续计算）
    net.send_data(conn, edge_output, "edge output")

    # 避免连续接收两个消息 防止消息粘包
    conn.sendall("avoid  sticky".encode())

    # 接收服务端返回的“物理极速时延”，但我们只做记录，不参与最终计算
    physical_latency = net.get_short_data(conn)

    # 4. 强制覆盖！使用我们计算出的模拟时延作为论文数据！
    transfer_latency = simulated_transfer_latency
    print(f"📦 包裹大小: {data_size / 1024 / 1024:.3f} MB")
    print(f"🚀 模拟设定带宽: {upload_bandwidth} MB/s")
    print(f"⏱️ {model_type} 模拟传输完成 - 耗时: {transfer_latency:.3f} ms (本地物理耗时仅: {physical_latency:.3f} ms)")
    # 避免连续接收两个消息 防止消息粘包
    conn.sendall("avoid  sticky".encode())

    cloud_latency, _ = net.get_data(conn)
    print(f"{model_type} 在云端设备上推理完成 - {cloud_latency:.3f} ms")
    actual_t_total = edge_latency + transfer_latency + cloud_latency
    print(
        f"================= 总耗时 T_total: {actual_t_total:.3f} ms =================")
    conn.close()
    return edge_latency,transfer_latency,cloud_latency


