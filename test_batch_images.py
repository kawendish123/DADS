import os
import torch
import cv2
import numpy as np
import time
import csv  # 引入 csv 模块
from datetime import datetime
from server_func import start_client
import warnings

warnings.filterwarnings("ignore")


def preprocess_image(img_path):
    """按照论文标准预处理单张图片"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)


def run_batch_test(img_dir, model_type, ip, port, bandwidth, Q, device,edge_only,cloud_only, num_frames=5):
    print(f"\n{'=' * 20} DADS 论文标准测试启动 {'=' * 20}")
    # 判断当前运行模式
    mode_str = "DADS (Dynamic Adaptive)"
    if edge_only:
        mode_str = "Edge-Only (纯边缘计算)"
    elif cloud_only:
        mode_str = "Cloud-Only (纯云端计算)"

    print(f"当前策略: {mode_str}")
    print(f"模型架构: {model_type} | 目标帧率 Q: {Q} FPS | 模拟网络带宽: {bandwidth} MB/s")

    # 根据当前实验参数和时间生成唯一的文件名
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{model_type}_Q{Q}_B{bandwidth}_{'edge_only' if edge_only else 'cloud_only' if cloud_only else 'dads'}_{current_time}.csv"

    # 创建并打开 CSV 文件，写入表头
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 表头：你可以根据毕业论文的需要增加更多列
        writer.writerow(["Frame ID", "Image Name", "Latency (ms)",
                         "edge_latency(ms)","transfer_latency(ms)","cloud_latency(ms) " "Target Q (FPS)", "Bandwidth (MB/s)"])
    # ---------------------------------------------------------

    img_names = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))][:num_frames]
    if not img_names:
        print("错误：未在文件夹中找到图片！")
        return

    latencies = []
    start_wall_time = time.time()

    for i, name in enumerate(img_names):
        img_path = os.path.join(img_dir, name)
        x = preprocess_image(img_path)
        if x is None: continue
        x = x.to(device)

        print(f"\n>>> 处理第 {i + 1}/{num_frames} 帧: {name}")

        # 核心：调用协同推理
        edge_latency,transfer_latency,cloud_latency = start_client(ip, port,
                                     x, model_type,
                                     bandwidth, device,Q,edge_only,cloud_only)
        frame_latency = edge_latency+transfer_latency+cloud_latency
        latencies.append(frame_latency)

        # ---------------------------------------------------------
        # 2. 将当前帧的结果实时追加到表格中
        # ---------------------------------------------------------
        with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入对应的数据行
            writer.writerow([i + 1, name, round(frame_latency, 3),
                             round(edge_latency, 3),round(transfer_latency, 3),round(cloud_latency, 3), Q, bandwidth])
        # ---------------------------------------------------------

    total_run_time = time.time() - start_wall_time
    avg_latency = np.mean(latencies)
    actual_throughput = 1000.0 / avg_latency if avg_latency > 0 else 0


    with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([])  # 空一行
        writer.writerow(["--- Summary ---"])
        writer.writerow(["Total Frames", len(latencies)])
        writer.writerow(["Average Latency (ms)", round(avg_latency, 2)])
        writer.writerow(["Throughput (FPS)", round(actual_throughput, 3)])
        writer.writerow(["Total Time (s)", round(total_run_time, 2)])
    # ---------------------------------------------------------

    print(f"\n{'=' * 20} 测试结果总结 {'=' * 20}")
    print(f"平均单帧延迟 (Latency): {avg_latency:.3f} ms")
    print(f"系统吞吐量 (Throughput): {actual_throughput:.3f} FPS")
    print(f"脚本测试总耗时: {total_run_time:.2f} s")
    print(f"\n✅ 数据已成功保存到表格文件: {csv_filename}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    run_batch_test(
        img_dir="./datasets/bdd100k/images/100k/apple",
        model_type="lite_hrnet",
        ip="127.0.0.1",
        port=9999,
        bandwidth=1000,
        Q=10,
        device="cpu",

        #纯边缘设备
        # edge_only=True,
        # cloud_only=False
        # #纯云设备
        edge_only=False,
        cloud_only=True
        # dads
        # edge_only=False,
        # cloud_only=False
    )