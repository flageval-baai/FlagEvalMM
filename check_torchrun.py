import os
import torch
import torch.distributed as dist
import socket


def main():
    # 1. 从环境变量获取分布式信息 (torchrun 会自动设置这些)
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]
        print(
            f"Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}, Master Addr: {master_addr}, Master Port: {master_port}"
        )
    except KeyError as e:
        print(f"Error: 环境变量缺失 {e}. 请确保使用 'torchrun' 启动此脚本。")
        return

    print(
        f"[Init] Rank {rank}/{world_size} (Local {local_rank}) on host {socket.gethostname()} starting..."
    )
    print(f"       Connecting to Master: {master_addr}:{master_port}")

    # 2. 设置当前进程使用的 GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(
            f"[Device] Rank {rank} is using GPU: {torch.cuda.get_device_name(local_rank)}"
        )
    else:
        print(f"[Device] Rank {rank} ERROR: No CUDA device found!")
        return

    # 3. 初始化进程组 (这是最容易卡住的一步)
    # 设置 timeout 为 30秒，防止无限期卡死
    try:
        dist.init_process_group(backend="nccl", init_method="env://")
        print(f"[Succ] Rank {rank} - Process Group Initialized!")
    except Exception as e:
        print(f"[Fail] Rank {rank} - Init Failed: {e}")
        raise e

    # 4. 执行一个简单的 NCCL 通信测试 (All-Reduce)
    # 创建一个张量，数值为 1
    tensor = torch.ones(1).to(device)

    print(f"[Comm] Rank {rank} starting all_reduce...")
    # 执行 all_reduce (默认是求和)，如果有 2 个卡，结果应该是 2
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    result = tensor.item()
    if result == world_size:
        print(f"[Done] Rank {rank} check PASSED! Result: {result} matches world_size.")
    else:
        print(
            f"[Fail] Rank {rank} check FAILED! Result: {result}, Expected: {world_size}"
        )

    # 5. 销毁进程组
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
