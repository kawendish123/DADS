from dads_framework.dinic import dinic_algorithm,get_min_cut_set
from dads_framework.graph_construct import graph_construct, get_transmission_latency_list
import numpy as np

def algorithm_DSL(model, model_input, edge_latency_list, cloud_latency_list, bandwidth, net_type="wifi"):
    """
    在低负载情况下为传入模型选择最优分割策略
    :param model: 传入DNN模型
    :param model_input: 模型输入
    :param edge_latency_list: 边缘设备上各层的推理时延
    :param cloud_latency_list: 云端设备上各层的推理时延
    :param bandwidth: 网络带宽 MB/s
    :param net_type: 当前网络带宽状况，默认为 "wifi"
    :return: 有向图中的对应的割集（不包含edge顶点和cloud顶点）以及划分过程会用到的 dict_node_layer，记录了顶点对应了第几层
    """
    # 构建对应的有向图
    graph, dict_node_layer, dict_layer_input_size = graph_construct(model, model_input, edge_latency_list, cloud_latency_list, bandwidth=bandwidth, net_type=net_type)
    # min_cut_value表示最短推理时延，reachable表示需要放在边缘端推理的顶点， non_reachable表示放在云端推理的顶点
    min_cut_value, reachable, non_reachable = dinic_algorithm(graph)

    # 检查一些bug时可能用到
    # for edge in graph.edges(data=True):
    #     print(edge)
    # print(reachable)
    # print(non_reachable)

    # partition_edge表示图中需要切割的边
    graph_partition_edge = get_min_cut_set(graph, min_cut_value, reachable, non_reachable)
    trans_lats = get_transmission_latency_list(model, model_input, bandwidth, net_type)
    te, tt, tc = get_physical_latencies(reachable, graph_partition_edge, dict_node_layer,
                                        edge_latency_list, cloud_latency_list, trans_lats)
    return graph_partition_edge,dict_node_layer,te, tt, tc


def algorithm_DSH(model, model_input, edge_lats, cloud_lats, bandwidth,
                  net_type="wifi", epsilon=0.01, K=2):
    """
    DSH 算法主程序：迭代精细化搜索最优切分方案
    """
    # 1. 预计算传输延迟列表 Ft
    trans_lats = get_transmission_latency_list(model, model_input, bandwidth, net_type)

    # 2. 设定搜索上限 (根据论文公式)
    min_ft = min([x for x in trans_lats if x > 0]) + 1e-5
    alpha_u = sum(edge_lats) / min_ft
    gamma_u = sum(cloud_lats) / min_ft

    print("\n>>> [数据诊断] <<<")
    print(f"边缘端总耗时 sum(Te): {sum(edge_lats):.4f}ms")
    print(f"云端总耗时   sum(Tc): {sum(cloud_lats):.4f}ms")
    print(f"最小传输耗时 min(Ft): {min_ft:.6f}ms")
    print("==========================================\n")

    # 搜索参数初始化
    current_space = [0, alpha_u, 0, gamma_u]  # [a_l, a_u, g_l, g_u]
    # 建议的初始化方式
    delta = max(alpha_u, gamma_u) / 5.0  # 初始切分 5 份进行粗扫
    # delta = 1  # 初始切分 5 份进行粗扫
    t_max_best = float('inf')
    final_config = None

    iteration = 1  # 迭代计数器

    print("-" * 50)
    print(f"DSH 搜索启动 | 初始 alpha_u: {alpha_u:.2f} | 初始 gamma_u: {gamma_u:.2f} |  目标: 最小化 max(Te, Tt, Tc)")
    print("-" * 50)

    # 3. 开始迭代优化
    while True:
        # 执行本轮搜索
        res = search_best_weights(
            model, model_input,
            (current_space[0], current_space[1]),
            (current_space[2], current_space[3]),
            delta, edge_lats, cloud_lats, trans_lats, bandwidth, net_type
        )

        # 2. 打印本轮的 final_config (res) 详细内容
        print(f"[迭代 {iteration}] 步长(delta): {delta:.3f}")
        print(f"  当前权重: alpha={res['alpha']:.3f}, gamma={res['gamma']:.3f}")
        print(f"  --> 当前瓶颈 (T_max): {res['t_max']:.4f}ms")
        print("-" * 30)

        improvement = t_max_best - res["t_max"]
        t_max_best = res["t_max"]
        final_config = res

        # 4. 终止条件：性能提升不明显或精度已足够
        if improvement < epsilon or delta < 0.05:
            break

        # 5. 缩小搜索空间：以当前最优解为中心，步长减半
        delta = delta / K
        current_space = [
            max(0, final_config["alpha"] - delta * K),
            final_config["alpha"] + delta * K,
            max(0, final_config["gamma"] - delta * K),
            final_config["gamma"] + delta * K
        ]
        iteration += 1

    print(f"DSH 搜索完成! 最优 T_max: {t_max_best:.4f}ms")
    print(f"最优 alpha: {final_config['alpha']:.4f}, 最优 gamma: {final_config['gamma']:.4f}")
    print(f"最优 partition: {final_config['partition']}")
    # print(f"最优 dict_node_layer: {final_config['dict_node_layer']}")
    return final_config["partition"], final_config["dict_node_layer"],t_max_best


def algorithm_DADS(model, model_input, edge_lats, cloud_lats, bandwidth, Q, net_type="wifi"):
    """
    DADS 主入口
    :param Q: Sampling Rate (帧率)，例如 30。1/Q 代表系统要求的单帧最大处理时间上限。
    """
    limit_ms = (1.0 / Q) * 1000.0
    print("\n" + "=" * 50)
    print(f"🚀 启动 DADS 动态调度 | 当前要求帧率 Q={Q} FPS, 瓶颈上限 {limit_ms:.2f}ms")
    print("=" * 50)

    # 1. 首先假设系统处于轻负载，运行 DSL
    print(">>> 阶段 1：尝试 DSL (追求最低总延迟)...")
    graph_edges, dict_layer, te, tt, tc = algorithm_DSL(model, model_input, edge_lats, cloud_lats, bandwidth, net_type)
    current_max_stage = max(te, tt, tc)


    # 2. 检查 DSL 方案是否会导致系统拥堵
    if current_max_stage > limit_ms:
        print(f">>> ⚠️ DSL 瓶颈耗时 {current_max_stage:.4f}ms > 限制 {limit_ms:.2f}ms，系统将面临拥堵！")
        print(">>> 阶段 2：切换至 DSH (追求最大吞吐量)...")

        graph_edges, dict_layer, t_max_dsh = algorithm_DSH(model, model_input, edge_lats, cloud_lats, bandwidth,
                                                           net_type)

        # 3. 检查即使是 DSH 是否也无能为力
        if t_max_dsh > limit_ms:
            print(
                f">>> 🚨 严重警告：DSH 最优瓶颈为 {t_max_dsh:.4f}ms依然无法满足 {Q} FPS。建议系统降级帧率 (inform-decrease)！")
    else:
        print(f">>> ✅ DSL 方案满足要求 (最大阶段耗时 {current_max_stage:.4f}ms)，采用 DSL 策略。")

    return graph_edges, dict_layer

def search_best_weights(model, model_input, alpha_range, gamma_range, delta,
                        edge_lats_orig, cloud_lats_orig, trans_lats_orig,
                        bandwidth, net_type):
    """
    在指定空间内通过 np.arange 遍历权重，寻找最优吞吐量配置
    """
    alpha_l, alpha_u = alpha_range
    gamma_l, gamma_u = gamma_range

    best_t_max = float('inf')
    best_round_config = {"alpha": 1.0, "gamma": 1.0, "partition": None, "dict_node_layer": None}

    # 双重循环遍历权重空间
    for a in np.arange(max(0.01, alpha_l), alpha_u + delta * 0.1, delta):
        for g in np.arange(max(0.01, gamma_l), gamma_u + delta * 0.1, delta):

            # A. 生成加权延迟（用于诱导算法切分）
            w_edge = [lat * a for lat in edge_lats_orig]
            w_cloud = [lat * g for lat in cloud_lats_orig]

            # B. 执行图构建与 Dinic 算法 (这里展开 DSL 逻辑以获取 reachable 集合)
            graph, dict_node_layer, _ = graph_construct(model, model_input, w_edge, w_cloud, bandwidth, net_type)
            min_val, reachable, non_reachable = dinic_algorithm(graph)
            partition_edges = get_min_cut_set(graph, min_val, reachable, non_reachable)

            # C. 核心评估：计算该方案在真实物理环境下的最大阶段延迟
            te, tt, tc = get_physical_latencies(reachable, partition_edges, dict_node_layer,
                                                edge_lats_orig, cloud_lats_orig, trans_lats_orig)

            current_t_max = max(te, tt, tc)  # 吞吐量瓶颈

            # D. 更新本轮搜索的最优解
            if current_t_max < best_t_max:
                best_t_max = current_t_max
                best_round_config.update({
                    "t_max": current_t_max,
                    "alpha": a, "gamma": g,
                    "partition": partition_edges,
                    "dict_node_layer": dict_node_layer,
                    "reachable": reachable  # 保存用于最终执行
                })

    return best_round_config


def get_physical_latencies(reachable, graph_partition_edge, dict_node_layer,
                           edge_lats_orig, cloud_lats_orig, trans_lats_orig):
    """
    还原真实物理时延，计算 Te, Tt, Tc
    """
    te_real, tc_real, tt_real = 0.0, 0.0, 0.0

    # 1. 计算真实的 Te (边缘) 和 Tc (云端)
    for node, layer_idx in dict_node_layer.items():
        if node in ["edge", "cloud"] or layer_idx == 0:
            continue

        # 映射回原始列表索引 (layer_idx 从 1 开始)
        idx = layer_idx - 1

        # 根据节点在可达集（边缘侧）还是不可达集（云端侧）累加原始延迟
        if node in reachable:
            te_real += edge_lats_orig[idx]
        else:
            tc_real += cloud_lats_orig[idx]

    # 2. 计算真实的 Tt (传输)
    for u, v in graph_partition_edge:
        # 仅当剪断的是物理层间的“黑边”时才计入传输
        if u in dict_node_layer and v in dict_node_layer:
            start_layer_idx = dict_node_layer[u]
            if start_layer_idx > 0:
                tt_real += trans_lats_orig[start_layer_idx - 1]

    return te_real, tt_real, tc_real

def get_partition_points(graph_partition_edge, dict_node_layer):
    """
    根据有向图的割集 graph_partition_edge 转换成DNN模型切分点 model_partition_edge
    :param graph_partition_edge: 有向图的割集
    :param dict_node_layer: 有向图顶点与模型层的对应
    :return: model_partition_edge: 模型中在哪两层之间进行分割
    """
    model_partition_edge = []
    for graph_edge in graph_partition_edge:
        # 表示在DNN模型中的第 start_layer 层 - end_layer之间进行划分(也就是说在start_layer之后进行划分)
        start_layer = dict_node_layer[graph_edge[0]]
        end_layer = dict_node_layer[graph_edge[1]]
        model_partition_edge.append((start_layer, end_layer))
    if not model_partition_edge:
        # 如果列表为空，说明算法选择了全边缘 (Edge-Only) 或全云端 (Cloud-Only)
        print(">>> 判定结果：当前列表为空，算法决策为【不进行切分】。")
        print("    可能原因：当前带宽 B 下，传输延迟 T_t 过高或过低，不切分才是全局最优解。")
    else:
        print(f">>> 判定结果：检测到切分点 {model_partition_edge}")
    return model_partition_edge

