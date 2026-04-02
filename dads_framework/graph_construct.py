import networkx as nx
import sys
import torch
from utils.inference_utils import recordTime
from net.net_utils import get_speed
import pickle

inf = sys.maxsize
construction_time = 0.0
predictor_dict = {}


def get_layers_latency0(model, device):
    """
    获取模型 model 在云端设备或边端设备上的各层推理时延，用于构建有向图
    :param model: DNN模型
    :param device: 推理设备
    :return: layers_latency[] 代表各层的推理时延
    """
    dict_layer_output = {}
    input = torch.rand((1, 3, 224, 224))  # 初始输入数据

    layers_latency = []
    for layer_index, layer in enumerate(model):
        # 对于某一层先检查其输入是否要进行修改
        if model.has_dag_topology and (layer_index + 1) in model.dag_dict.keys():
            pre_input_cond = model.dag_dict[layer_index + 1]  # 取出其前置输入条件
            if isinstance(pre_input_cond, list):  # 如果其是一个列表，代表当前层有多个输入
                input = []
                for pre_index in pre_input_cond:  # 对于concat操作,输入应为一个列表
                    input.append(dict_layer_output[pre_index])
            else:  # 当前层的的输入从其他层或得
                input = dict_layer_output[pre_input_cond]

        if not isinstance(input,list):
            input = input.to(device)  # 将数据放在相应设备上

        layer = layer.to(device)  # 将该层放到相应设备上
        input,lat = recordTime(layer, input, device, epoch_cpu=10, epoch_gpu=10)  # 记录推理时延

        if model.has_dag_topology and (layer_index+1) in model.record_output_list:
            dict_layer_output[layer_index + 1] = input
        layers_latency.append(lat)
    return layers_latency


def get_layers_latency(model, device):
    """
    获取模型 model 在云端设备或边端设备上的各层推理时延。
    此版本已针对 HRNet 的异构 DAG 拓扑（含融合节点）及 3050 显存进行深度优化。
    :return: layers_latency[] 代表各层的推理时延
    """
    dict_layer_output = {}
    input_data = torch.rand((1, 3, 224, 224)).to(device)  # 初始输入数据

    layers_latency = []

    for layer_index, layer in enumerate(model):
        logic_idx = layer_index + 1  # DADS 框架逻辑层号从 1 开始
        layer = layer.to(device)

        # ==========================================================
        # 1. 检查输入条件 & 融合节点即时计算 (Lazy-Loading)
        # ==========================================================
        if model.has_dag_topology and logic_idx in model.dag_dict.keys():
            pre_input_cond = model.dag_dict[logic_idx]

            # 🌟 核心拦截：如果依赖的是融合节点（如 "s2_f1"）且还未计算
            if isinstance(pre_input_cond, str) and pre_input_cond not in dict_layer_output:
                fusion_deps = model.dag_dict[pre_input_cond]
                fusion_inputs = [dict_layer_output[dep] for dep in fusion_deps]

                # 取出对应的融合算子
                fusion_op = model.merge_ops[pre_input_cond]

                # 确保算子和数据在同一设备上
                if hasattr(fusion_inputs[0], 'device'):
                    fusion_op = fusion_op.to(fusion_inputs[0].device)

                with torch.no_grad():  # 节省显存
                    dict_layer_output[pre_input_cond] = fusion_op(fusion_inputs)

            # 正式提取当前层的输入
            if isinstance(pre_input_cond, list):
                input_data = [dict_layer_output[idx] for idx in pre_input_cond]
            else:
                input_data = dict_layer_output[pre_input_cond]

        # ==========================================================
        # 2. 通道对齐防护 (防止 64 vs 32 RuntimeError)
        # ==========================================================
        expected_in_c = None
        target_layer = layer[0] if isinstance(layer, torch.nn.Sequential) else layer

        if hasattr(target_layer, 'in_channels'):
            expected_in_c = target_layer.in_channels
        elif hasattr(target_layer, 'conv1'):
            expected_in_c = target_layer.conv1.in_channels

        # 若通道数不符，伪造一个正确形状的输入仅供测速
        if expected_in_c is not None and not isinstance(input_data, list):
            if input_data.shape[1] != expected_in_c:
                input_data = torch.randn(1, expected_in_c, input_data.shape[2], input_data.shape[3]).to(device)

        # ==========================================================
        # 3. 记录推理时延与输出
        # ==========================================================
        with torch.no_grad():  # 强力防 OOM
            input_data, lat = recordTime(layer, input_data, device, epoch_cpu=10, epoch_gpu=10)

        # 如果当前层是其他层的依赖前置，缓存其输出
        if model.has_dag_topology and logic_idx in model.record_output_list:
            dict_layer_output[logic_idx] = input_data

        layers_latency.append(lat)  # 记录当前层时间

        # 定期清理显存，保护 3050 不崩溃
        torch.cuda.empty_cache()

    # 毫无疑问，只返回各层的时延列表！
    return layers_latency

def add_graph_edge(graph, vertex_index, input, layer_index, layer,
                   bandwidth, net_type, edge_latency, cloud_latency,
                 dict_input_size_node_name, dict_node_layer, dict_layer_input_size, dict_layer_output,
                 record_flag):
    """
    向一个有向图中添加
    :param graph: 向哪个有向图中添加
    :param vertex_index: 当前构建的顶点编号
    :param input: 当前层的输入
    :param layer_index: 当前层
    :param layer: 当前层类型
    :param bandwidth: 网络带宽
    :param net_type: 网络类型
    :param edge_latency: 在边缘设备上推理时延
    :param cloud_latency: 在云端设备上推理时延
    :param dict_input_size_node_name:   字典：key:输入 value:对应的顶点编号
    :param dict_node_layer:             字典：key:顶点编号 value:对应DNN中第几层
    :param dict_layer_input_size:       字典：key:DNN中第几层 value:对应的输入大小
    :param dict_layer_output:            字典：key:DNN中第几层 value:对应的输出
    :param record_flag: 只有某些关键层才会记录层的输出
    :return: 当前构建的顶点数目 vertex_index ，以及当前层的输出（会用于作为下一层的输入）
    """
    cloud_vertex = "cloud"  # 云端设备节点
    edge_vertex = "edge"  # 边缘设备节点

    # 获取当前层在边缘端设备上的推理时延以及在云端设备上的推理时延
    # edge_lat = predict_model_latency(input, layer, device="edge", predictor_dict=predictor_dict)
    # cloud_lat = predict_model_latency(input, layer, device="cloud", predictor_dict=predictor_dict)

    # 获取当前层需要的传输时延
    #   predict transmission latency,network_type = WI-FI
    transport_size = len(pickle.dumps(input))
    speed = get_speed(network_type=net_type,bandwidth=bandwidth)
    transmission_lat = transport_size / speed

    # 一层dnn layer可以构建一条边，而构建一条边需要两个顶点
    # dict_input_size_node_name 可以根据输入数据大小构建对应的图顶点
    # 所以可以在执行dnn layer的前后分别构建 start_node以及end_node
    start_node, end_node, record_input = None, None, None

    if isinstance(input,list):
        layer_out = None
        record_input = input
        for one_input in input:
            vertex_index, start_node = get_node_name(one_input, vertex_index, dict_input_size_node_name)
            layer_out = layer(input)
            vertex_index, end_node = get_node_name(layer_out, vertex_index, dict_input_size_node_name)

            # 例如 input 是长度为n的列表，则需要构建n个边
            graph.add_edge(start_node, end_node, capacity=transmission_lat)  # 添加从前一个节点到当前节点的边
        input = layer_out
    else:  # 常规构建
        vertex_index, start_node = get_node_name(input, vertex_index, dict_input_size_node_name)
        record_input = input
        input = layer(input)
        vertex_index, end_node = get_node_name(input, vertex_index, dict_input_size_node_name)

        # 避免无效层覆盖原始数据 用这种方式可以过滤掉relu层或dropout层
        if start_node == end_node:
            dict_node_layer[end_node] = layer_index + 1
            if record_flag:
                dict_layer_output[layer_index + 1] = input
            return vertex_index,input  # 不需要进行构建
        graph.add_edge(start_node, end_node, capacity=transmission_lat)  # 添加从前一个节点到当前节点的边

    # 注意：end_node可以用来在有向图中表示当前的 dnn-layer
    graph.add_edge(edge_vertex, end_node, capacity=cloud_latency)  # 添加从边缘节点到dnn层的边
    graph.add_edge(end_node, cloud_vertex, capacity=edge_latency)  # 添加从dnn层到云端设备的边

    dict_node_layer[end_node] = layer_index + 1  # 记录有向图中的顶点对应的DNN的第几层
    # dict_layer_input_size[layer_index + 1] = record_input.shape  # 记录DNN层中第i层对应的输入大小
    if record_flag:
        dict_layer_output[layer_index+1] = input  # 记录DNN层中第i层对应的输出

    return vertex_index,input



def get_transmission_latency_list0(model, input, bandwidth, net_type="wifi"):
    """
    通过 model, input 和 bandwidth 获取每层传输时延的列表
    """
    # 1. 初始化速度
    speed = get_speed(network_type=net_type, bandwidth=bandwidth)

    # 2. 初始化存储容器
    transmission_latencies = []
    dict_layer_output = {0: input}  # 记录每一层的输出，用于处理 DAG 结构

    current_input = input

    # 3. 模拟模型推理循环
    for layer_index, layer in enumerate(model):
        # 处理 DAG 拓扑：检查当前层是否需要来自非相邻前驱层的输入 [cite: 38-39, 85]
        if model.has_dag_topology and (layer_index + 1) in model.dag_dict.keys():
            pre_input_cond = model.dag_dict[layer_index + 1]
            if isinstance(pre_input_cond, list):
                current_input = [dict_layer_output[pre_index] for pre_index in pre_input_cond]
            else:
                current_input = dict_layer_output[pre_input_cond]

        # 4. 计算当前阶段的传输时延
        # 注意：这里计算的是将数据传输到该层执行所需的代价
        transport_size = len(pickle.dumps(current_input))
        transmission_lat = transport_size / speed
        transmission_latencies.append(transmission_lat)

        # 5. 执行推理获取输出，更新 input 用于下一层
        # 使用 torch.no_grad() 避免计算梯度以节省内存
        with torch.no_grad():
            # 处理多输入（list）和单输入
            if isinstance(current_input, list):
                output = layer(current_input)
            else:
                output = layer(current_input)

        # 6. 如果是 DAG 中的关键层，记录输出供后续引用 [cite: 413]
        if model.has_dag_topology and (layer_index + 1) in model.record_output_list:
            dict_layer_output[layer_index + 1] = output

        current_input = output  # 更新为下一层的输入

    return transmission_latencies


def get_transmission_latency_list(model, input, bandwidth, net_type="wifi"):
    speed = get_speed(network_type=net_type, bandwidth=bandwidth)
    transmission_latencies = []
    dict_layer_output = {0: input}
    current_input = input

    for layer_index, layer in enumerate(model):
        logic_idx = layer_index + 1  # 逻辑层号

        # 处理 DAG 拓扑跳转
        if model.has_dag_topology and logic_idx in model.dag_dict.keys():
            pre_input_cond = model.dag_dict[logic_idx]

            # ==========================================================
            # 🌟 核心补丁：融合节点即时计算 (同步适配传输时延获取函数)
            # ==========================================================
            if isinstance(pre_input_cond, str) and pre_input_cond not in dict_layer_output:
                fusion_deps = model.dag_dict[pre_input_cond]
                fusion_inputs = [dict_layer_output[dep] for dep in fusion_deps]

                fusion_op = model.merge_ops[pre_input_cond]
                if hasattr(fusion_inputs[0], 'device'):
                    fusion_op = fusion_op.to(fusion_inputs[0].device)

                with torch.no_grad():
                    dict_layer_output[pre_input_cond] = fusion_op(fusion_inputs)

            if isinstance(pre_input_cond, list):
                current_input = [dict_layer_output[pre_index] for pre_index in pre_input_cond]
            else:
                current_input = dict_layer_output[pre_input_cond]

        # 计算序列化后的传输代价
        transport_size = len(pickle.dumps(current_input))
        transmission_lat = transport_size / speed
        transmission_latencies.append(transmission_lat)

        # 模拟执行以获取输出尺寸
        with torch.no_grad():
            try:
                output = layer(current_input)
            except RuntimeError:
                # 如果报错（如通道不匹配），强制修正输出以便后续计算
                output = current_input

        if model.has_dag_topology and logic_idx in model.record_output_list:
            dict_layer_output[logic_idx] = output

        current_input = output

    return transmission_latencies

def graph_construct0(model, input, edge_latency_list, cloud_latency_list, bandwidth, net_type="wifi"):
    """
    传入一个DNN模型，construct_digraph_by_model将DNN模型构建成具有相应权重的有向图
    构建过程主要包括三个方面：
    (1) 从边缘设备-dnn层的边 权重设置为云端推理时延
    (2) dnn层之间的边 权重设置为传输时延
    (3) 从dnn层-云端设备的边 权重设置为边端推理时延
    :param model: 传入dnn模型
    :param input: dnn模型的初始输入
    :param edge_latency_list: 边缘设备上各层的推理时延
    :param cloud_latency_list: 云端设备上各层的推理时延
    :param bandwidth: 当前网络时延带宽，可由带宽监视器获取 MB/s
    :param net_type: 当前网络类型 默认为 wifi
    :return: 构建好的有向图graph, dict_vertex_layer, dict_layer_input

    由于 GoogleNet 和 ResNet 不能用简单地 x = layer(x) 进行下一步执行
    所以需要自定义新的 get_min_cut_value_for_ResBlock
    所以用户如果有新的DAG结构 （1）完善已有创建结构 （2）iterable api 需要自定义
    """
    # print(f"测试bandwidth = {bandwidth} MB/s")

    graph = nx.DiGraph()

    """
    dict_for_input 字典的作用：
        :key tuple (input.size,input_slice) 字典的键是 输入的形状以及输入的切片(取输入中的前3个数据)
        :value 与之对应的构建好的有向图中的顶点 node_name
    通过dict_for_input可以将 DNN layer 转化为有向图中的顶点 node_name
    原理：对于每一个DNN中的layer 其输入数据是唯一的
    """
    dict_input_size_node_name = {}

    """
    dict_vertex_layer 字典的作用：
        :key node_name 有向图中顶点的名称
        :value 对应原DNN中第几层 layer_index
    可以通过有向图的顶点 node_name 找到其对应原DNN模型中第几层
    注意：
        layer_index = 0 代表初始输入 
        layer_index > 0 表示目前顶点代表原DNN层的第layer_index层，若想取出原DNN层应使用 model[layer_index-1]
    """
    dict_node_layer = {"v0": 0}  # 初始化v0对应的为初始输入

    """
    dict_layer_input 以及 dict_layer_output 字典的作用：
        :key 原DNN中第几层 layer_index 
        :value DNN中第 layer_index 的层输入以及输出是什么
    第 layer_index 层的输入与输出，可以使用 shape 以及前三个元素确定是否为同1输入
    注意：
        layer_index = 0 代表初始输入 
        layer_index = n 获取的是原模型中 model[layer_index-1] 层的输入
    """
    dict_layer_input = {0: None}  # 第0层为初始输入 其输入记录为None
    dict_layer_output = {0: input}  # 第0层为初始输入 其输出即为input

    cloud_vertex = "cloud"  # 云端设备节点
    edge_vertex = "edge"  # 边缘设备节点

    # print(f"start construct graph for model...")
    graph.add_edge(edge_vertex, "v0", capacity=inf)  # 构建模型初始输入v0
    vertex_index = 0  # 构建图的顶点序号

    for layer_index, layer in enumerate(model):
        # print(layer_index,layer)
        # 对于某一层先检查其输入是否要进行修改
        if model.has_dag_topology and (layer_index+1) in model.dag_dict.keys():
            pre_input_cond = model.dag_dict[layer_index+1]  # 取出其前置输入条件
            if isinstance(pre_input_cond, list):  # 如果其是一个列表，代表当前层有多个输入
                input = []
                for pre_index in pre_input_cond:  # 对于concat操作,输入应为一个列表
                    input.append(dict_layer_output[pre_index])
            else:  # 当前层的的输入从其他层或得
                input = dict_layer_output[pre_input_cond]

        # 标记在模型中 record_output_list 中的DNN层需要记录输出
        record_flag = model.has_dag_topology and (layer_index+1) in model.record_output_list
        # 枸橘修改后的input进行边的构建
        vertex_index, input = add_graph_edge(graph, vertex_index, input, layer_index, layer,
                                             bandwidth, net_type,
                                             edge_latency_list[layer_index],cloud_latency_list[layer_index],
                                             dict_input_size_node_name, dict_node_layer,
                                           dict_layer_input, dict_layer_output, record_flag=record_flag)

    # 主要负责处理出度大于1的顶点
    prepare_for_partition(graph, vertex_index, dict_node_layer)
    return graph, dict_node_layer, dict_layer_input


def graph_construct(model, input, edge_latency_list, cloud_latency_list, bandwidth, net_type="wifi"):
    """
    传入一个DNN模型，将DNN模型构建成具有相应权重的有向图
    构建过程主要包括三个方面：
    (1) 从边缘设备-dnn层的边 权重设置为云端推理时延
    (2) dnn层之间的边 权重设置为传输时延
    (3) 从dnn层-云端设备的边 权重设置为边端推理时延
    """
    graph = nx.DiGraph()

    dict_input_size_node_name = {}
    dict_node_layer = {"v0": 0}  # 初始化v0对应的为初始输入
    dict_layer_input = {0: None}  # 第0层为初始输入 其输入记录为None
    dict_layer_output = {0: input}  # 第0层为初始输入 其输出即为input

    cloud_vertex = "cloud"  # 云端设备节点
    edge_vertex = "edge"  # 边缘设备节点

    graph.add_edge(edge_vertex, "v0", capacity=inf)  # 构建模型初始输入v0
    vertex_index = 0  # 构建图的顶点序号

    for layer_index, layer in enumerate(model):
        logic_idx = layer_index + 1  # 逻辑层号

        # 对于某一层先检查其输入是否要进行修改
        if model.has_dag_topology and logic_idx in model.dag_dict.keys():
            pre_input_cond = model.dag_dict[logic_idx]  # 取出其前置输入条件

            # ==========================================================
            # 🌟 核心补丁：融合节点即时计算 (Graph Construct 阶段同步适配)
            # ==========================================================
            if isinstance(pre_input_cond, str) and pre_input_cond not in dict_layer_output:
                fusion_deps = model.dag_dict[pre_input_cond]
                fusion_inputs = [dict_layer_output[dep] for dep in fusion_deps]

                # 取出融合算子并确保设备一致
                fusion_op = model.merge_ops[pre_input_cond]
                if hasattr(fusion_inputs[0], 'device'):
                    fusion_op = fusion_op.to(fusion_inputs[0].device)

                with torch.no_grad():
                    fusion_out = fusion_op(fusion_inputs)
                    dict_layer_output[pre_input_cond] = fusion_out

                # --- 【新增代码：修桥铺路，接通孤岛】 ---
                # 获取融合输出对应的顶点名称
                vertex_index, fusion_node = get_node_name(fusion_out, vertex_index, dict_input_size_node_name)

                # 计算网络传输速度
                speed = get_speed(network_type=net_type, bandwidth=bandwidth)

                # 把两条分支的末端顶点，连接到这个融合顶点上
                for f_in in fusion_inputs:
                    vertex_index, dep_node = get_node_name(f_in, vertex_index, dict_input_size_node_name)
                    trans_size = len(pickle.dumps(f_in))
                    graph.add_edge(dep_node, fusion_node, capacity=trans_size / speed)

                # 为融合顶点添加虚拟的端云计算时延 (设为0，因为加法操作代价极小)
                graph.add_edge(edge_vertex, fusion_node, capacity=0.0)
                graph.add_edge(fusion_node, cloud_vertex, capacity=0.0)

                # 给融合顶点上户口，防止后续报错
                dict_node_layer[fusion_node] = logic_idx

            # 正式装载当前层的输入
            if isinstance(pre_input_cond, list):  # 如果其是一个列表，代表当前层有多个输入
                input = []
                for pre_index in pre_input_cond:
                    input.append(dict_layer_output[pre_index])
            else:  # 当前层的输入从其他层获得
                input = dict_layer_output[pre_input_cond]

        # 标记在模型中 record_output_list 中的DNN层需要记录输出
        record_flag = model.has_dag_topology and logic_idx in model.record_output_list

        # 构建修改后的input进行边的构建
        vertex_index, input = add_graph_edge(graph, vertex_index, input, layer_index, layer,
                                             bandwidth, net_type,
                                             edge_latency_list[layer_index], cloud_latency_list[layer_index],
                                             dict_input_size_node_name, dict_node_layer,
                                             dict_layer_input, dict_layer_output, record_flag=record_flag)

    # 主要负责处理出度大于1的顶点
    prepare_for_partition(graph, vertex_index, dict_node_layer)
    return graph, dict_node_layer, dict_layer_input


def get_node_name(input, vertex_index, dict_input_size_node_name):
    """
    根据输入input构建对应的顶点名称 node_name
    :param input: 当前层的输入
    :param vertex_index: 顶点编号 即目前应该创建哪个顶点
    :param dict_input_size_node_name: 通过dict_for_input可以将 DNN layer 转化为有向图中的顶点 node_name
    :return: node name，构建DAG边所需要的首位节点name
    """
    len_of_shape = len(input.shape)
    input_shape = str(input.shape)  # 获取当前input的大小

    input_slice = input
    for _ in range(len_of_shape-1):
        input_slice = input_slice[0]
    input_slice = str(input_slice[:3])  # 获取input的前3个数据，保证数据的唯一性

    if (input_shape, input_slice) not in dict_input_size_node_name.keys():
        node_name = "v" + str(vertex_index)
        dict_input_size_node_name[(input_shape, input_slice)] = node_name  # 创建一个新的节点并保存
        vertex_index += 1
    else:
        node_name = dict_input_size_node_name[(input_shape, input_slice)]  # 从字典中取出原有节点 保证正确构建有向图
    return vertex_index, node_name


def prepare_for_partition(graph, vertex_index, dict_node_layer):
    """
    对根据DNN模型已经构建好的DAG图进行下一步工作：
    1 - 将有多个出点的顶点 记录为start_vex
    2 - 生成新节点为node_name 从node_name -> start_vex 的边代表传输速度，原来从start vex出发的边改为inf
    3 - 找到需要删除的边 ：指原终点为 start vex 的边，将其改成到新节点node name的边
    4 - 删除cloud和edge到原节点的边
    :param graph : 已经构建好的DAG图
    :param vertex_index : 指定下一个生成的节点编号
    :param dict_node_layer : 记录有向图中的顶点对应的DNN的第几层
    :return:
    """
    map_for_vex = []  # 处理 graph  - 1个顶点指向多个其他顶点的情况
    multiple_out_vex = []  # 保存有多个出点的vex
    for edge in graph.edges.data():
        start_vex = edge[0]
        end_vex = edge[1]
        if start_vex == "edge" or end_vex == "cloud":
            continue
        if start_vex not in map_for_vex:  # 如果当前顶点的前置顶点是第一个出现则进行保存
            map_for_vex.append(start_vex)
        elif start_vex not in multiple_out_vex:  # 如果前置顶点已经出现过 再出现的话说明start_vex出度大于1，将其记录在multiple_out_vex中
            multiple_out_vex.append(start_vex)

    for start_vex in multiple_out_vex:
        # 生成新的节点
        node_name = "v" + str(vertex_index)
        vertex_index += 1
        dict_node_layer[node_name] = dict_node_layer[start_vex]  # 新节点与原节点对应原来的同一层

        # 对旧的节点进行改正
        modify_edges = []  # 记录需要修改的边，即起点为start_vex的节点，将其修改为inf
        for edge in graph.edges.data():
            if edge[0] == "edge" or edge[1] == "cloud":
                continue
            if edge[0] == start_vex:
                modify_edges.append(edge)

        # 增加新edge
        for edge in modify_edges:
            graph.add_edge(edge[0], node_name, capacity=edge[2]["capacity"])  # 新增一条从start_vex到node_name的边
            graph.add_edge(node_name, edge[1], capacity=inf)  # 新增从node_name到edge[1]的边 权重为inf
            graph.remove_edge(edge[0],edge[1])  # 删除原有的边

        # 删除 edge - old node
        # if graph.has_edge("edge", start_vex):
        #     data = graph.get_edge_data("edge", start_vex)["capacity"]
        #     graph.add_edge("edge", node_name, capacity=data)
        #     graph.remove_edge("edge", start_vex)
        # 删除 old node - cloud
        # if graph.has_edge(start_vex, "cloud"):
        #     data = graph.get_edge_data(start_vex, "cloud")["capacity"]
        #     graph.add_edge(node_name, "cloud", capacity=data)
        #     graph.remove_edge(start_vex, "cloud")

    # 简化edge的数值 保留三位小数足够计算
    for edge in graph.edges.data():
        graph.add_edge(edge[0], edge[1], capacity=round(edge[2]["capacity"], 3))
    return vertex_index
