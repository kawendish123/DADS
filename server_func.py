from utils import inference_utils
from dads_framework.dads import algorithm_DSL, get_partition_points, algorithm_DSH, algorithm_DADS
from dads_framework.graph_construct import get_layers_latency
import net.net_utils as net

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
    net.send_short_data(conn, cloud_latency_list, "model latency on the cloud device.")

    # 接收模型分层点
    model_partition_edge = net.get_short_data(conn)
    print(f"get partition point successfully.")

    # 获取划分后的边缘端模型和云端模型
    _, cloud_model = inference_utils.model_partition(model, model_partition_edge)
    cloud_model = cloud_model.to(device)

    # 接收中间数据并返回传输时延
    edge_output, transfer_latency = net.get_data(conn)

    # 避免连续发送两个消息 防止消息粘包
    conn.recv(40)

    print(f"get edge_output and transfer latency successfully.")
    net.send_short_data(conn, transfer_latency, "transfer latency")

    # 避免连续发送两个消息 防止消息粘包
    conn.recv(40)

    inference_utils.warmUp(cloud_model, edge_output, device)

    # 记录云端推理时延
    cloud_output, cloud_latency = inference_utils.recordTime(cloud_model, edge_output, device, epoch_cpu=30,
                                                             epoch_gpu=100)
    print(f"{model_type} 在云端设备上推理完成 - {cloud_latency:.3f} ms")
    net.send_short_data(conn, cloud_latency, "cloud latency")

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
    cloud_latency_list = net.get_short_data(conn)  # 接受到云端的时延参数

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

    # 获取划分后的边缘端模型和云端模型
    edge_model, _ = inference_utils.model_partition(model, model_partition_edge)
    edge_model = edge_model.to(device)

    # 开始边缘端的推理 首先进行预热
    inference_utils.warmUp(edge_model, input_x, device)
    edge_output, edge_latency = inference_utils.recordTime(edge_model, input_x, device, epoch_cpu=30, epoch_gpu=100)
    print(f"{model_type} 在边缘端设备上推理完成 - {edge_latency:.3f} ms")

    # 发送中间数据
    net.send_data(conn, edge_output, "edge output")

    # 避免连续接收两个消息 防止消息粘包
    conn.sendall("avoid  sticky".encode())

    transfer_latency = net.get_short_data(conn)
    print(f"{model_type} 传输完成 - {transfer_latency:.3f} ms")

    # 避免连续接收两个消息 防止消息粘包
    conn.sendall("avoid  sticky".encode())

    cloud_latency = net.get_short_data(conn)
    print(f"{model_type} 在云端设备上推理完成 - {cloud_latency:.3f} ms")
    actual_t_total = edge_latency + transfer_latency + cloud_latency
    print(
        f"================= 总耗时 T_total: {actual_t_total:.3f} ms =================")
    conn.close()
    return edge_latency,transfer_latency,cloud_latency


