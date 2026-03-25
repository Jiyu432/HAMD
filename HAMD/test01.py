import torch
import time


def gpu_benchmark():
    size = 40000  # 根据实际显存调整矩阵大小

    # 确保有足够的GPU可用
    if torch.cuda.device_count() < 2:
        raise SystemError("This benchmark requires at least two GPUs")

    print("Using GPUs:", ", ".join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))

    # 创建两个大矩阵并分配到不同GPU
    a = torch.rand(size, size, device="cuda:0")
    b = torch.rand(size, size, device="cuda:1")

    # 将所有数据移动到 cuda:0
    b = b.to('cuda:0')

    try:
        start_time = time.time()
        # 在cuda:0上执行矩阵乘法
        c = torch.matmul(a, b)
        torch.cuda.synchronize()  # 确保CUDA核心已经完成所有任务
        end_time = time.time()
        print(f"Matrix multiplication took {end_time - start_time:.5f} seconds")
    except KeyboardInterrupt:
        print("Benchmark stopped by user.")


if __name__ == "__main__":
    gpu_benchmark()
