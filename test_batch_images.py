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


def run_batch_test(img_dir, model_type, ip, port, bandwidth, Q, device, num_frames=10):
    print(f"\n{'=' * 20} DADS 论文标准测试启动 {'=' * 20}")
    print(f"模型: {model_type} | 目标帧率 Q: {Q} | 模拟带宽: {bandwidth} MB/s")

    # ---------------------------------------------------------
    # 1. 准备 CSV 表格文件
    # ---------------------------------------------------------
    # 根据当前实验参数和时间生成唯一的文件名
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"dads_result_{model_type}_Q{Q}_B{bandwidth}_{current_time}.csv"

    # 创建并打开 CSV 文件，写入表头
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 表头：你可以根据毕业论文的需要增加更多列
        writer.writerow(["Frame ID", "Image Name", "Latency (ms)", "Target Q (FPS)", "Bandwidth (MB/s)"])
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

        t_start = time.time()

        # 核心：调用协同推理
        start_client(ip, port, x, model_type, bandwidth, device, Q=Q)

        t_end = time.time()
        frame_latency = (t_end - t_start) * 1000  # 转换为毫秒 ms
        latencies.append(frame_latency)

        # ---------------------------------------------------------
        # 2. 将当前帧的结果实时追加到表格中
        # ---------------------------------------------------------
        with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入对应的数据行
            writer.writerow([i + 1, name, round(frame_latency, 2), Q, bandwidth])
        # ---------------------------------------------------------

        # 模拟真实帧间隔
        expected_interval = 1.0 / Q
        elapsed = t_end - t_start
        if elapsed < expected_interval:
            time.sleep(expected_interval - elapsed)

    total_run_time = time.time() - start_wall_time
    avg_latency = np.mean(latencies)
    throughput = len(latencies) / total_run_time

    # ---------------------------------------------------------
    # 3. 将最终的汇总数据也写在表格最下方
    # ---------------------------------------------------------
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([])  # 空一行
        writer.writerow(["--- Summary ---"])
        writer.writerow(["Total Frames", len(latencies)])
        writer.writerow(["Average Latency (ms)", round(avg_latency, 2)])
        writer.writerow(["Throughput (FPS)", round(throughput, 2)])
        writer.writerow(["Total Time (s)", round(total_run_time, 2)])
    # ---------------------------------------------------------

    print(f"\n{'=' * 20} 测试结果总结 {'=' * 20}")
    print(f"平均单帧延迟 (Latency): {avg_latency:.2f} ms")
    print(f"系统吞吐量 (Throughput): {throughput:.2f} FPS")
    print(f"测试总耗时: {total_run_time:.2f} s")
    print(f"\n✅ 数据已成功保存到表格文件: {csv_filename}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    run_batch_test(
        img_dir="./datasets/bdd100k/images/100k/apple",
        model_type="vgg_net",
        ip="127.0.0.1",
        port=9999,
        bandwidth=1.1,  # 模拟 3G 网络
        Q=20.0,  # 设置为高负载
        device="cpu"
    )