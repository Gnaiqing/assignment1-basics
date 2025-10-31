import argparse, os, time, json, platform, torch
from torch import nn


# 1. env dump ---------------------------------------------------------------
def dump_env():
    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_name_0"] = torch.cuda.get_device_name(0)
        info["capability_0"] = torch.cuda.get_device_capability(0)
        info["cuda_version"] = torch.version.cuda
        info["driver_version"] = torch.cuda.get_device_properties(0).driver_version \
            if hasattr(torch.cuda.get_device_properties(0), "driver_version") else "n/a"
    print("=== ENV ===")
    print(json.dumps(info, indent=2))


# 2. pure GPU matmul benchmark ----------------------------------------------
@torch.inference_mode()
def matmul_benchmark(device="cuda", iters=50, size=8192):
    # 8192x8192 matmul ~ good stress for H100; adjust if OOM
    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)
    # warmup
    for _ in range(5):
        (A @ B).sum().item()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        C = A @ B
    torch.cuda.synchronize()
    t1 = time.time()
    secs = t1 - t0
    flop = 2 * (size ** 3)  # matmul FLOPs
    tflops = (flop * iters) / secs / 1e12
    print(f"[MATMUL] size={size}, iters={iters}, time={secs:.3f}s, approx={tflops:.2f} TFLOP/s")
    return tflops


# 3. fake transformer forward+backward --------------------------------------
class TinyBlock(nn.Module):
    def __init__(self, d_model=2048, n_heads=16, seq_len=1024):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.seq_len = seq_len

    def forward(self, x):
        x = self.ln(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = x + attn_out
        x = x + self.ff(x)
        return x


def training_like_benchmark(device="cuda", bsz=8, seq_len=1024, d_model=2048, amp=True, steps=50):
    model = TinyBlock(d_model=d_model, n_heads=16, seq_len=seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    x = torch.randn(bsz, seq_len, d_model, device=device)
    target = torch.randn_like(x)

    # warmup
    for _ in range(5):
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(x)
            loss = (out - target).pow(2).mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(x)
            loss = (out - target).pow(2).mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t1 = time.time()
    step_time = (t1 - t0) / steps
    it_per_s = 1.0 / step_time
    print(f"[TRAIN] bsz={bsz}, seq={seq_len}, d_model={d_model}, amp={amp} -> {step_time*1000:.1f} ms/step ({it_per_s:.2f} it/s)")
    return step_time


# 4. dataloader / host→device test ------------------------------------------
def h2d_benchmark(device="cuda", bsz=32, num_batches=200, shape=(3, 1024, 1024)):
    # pretend we load from CPU tensors (what dataloader does)
    cpu_batch = torch.randn(bsz, *shape, pin_memory=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(num_batches):
        gpu_batch = cpu_batch.to(device, non_blocking=True)
        # do a tiny op so CUDA actually consumes it
        _ = gpu_batch * 2.0
    torch.cuda.synchronize()
    t1 = time.time()
    throughput = num_batches / (t1 - t0)
    print(f"[H2D] bsz={bsz}, shape={shape}, {throughput:.1f} batches/s")
    return throughput


# 5. disk / filesystem test -------------------------------------------------
def disk_benchmark(path="/tmp/bench.bin", size_gb=4):
    # write
    data = os.urandom(1024 * 1024)  # 1MB chunk
    total = size_gb * 1024  # MB
    t0 = time.time()
    with open(path, "wb") as f:
        for _ in range(total):
            f.write(data)
    t1 = time.time()
    write_speed = size_gb / (t1 - t0)
    # read
    t2 = time.time()
    with open(path, "rb") as f:
        while f.read(1024 * 1024):
            pass
    t3 = time.time()
    read_speed = size_gb / (t3 - t2)
    print(f"[DISK] wrote {size_gb}GB -> {write_speed:.2f} GB/s, read -> {read_speed:.2f} GB/s (path={path})")
    return write_speed, read_speed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--matmul-size", type=int, default=8192)
    parser.add_argument("--disk-path", type=str, default="/tmp/bench.bin")
    parser.add_argument("--skip-disk", action="store_true")
    args = parser.parse_args()

    dump_env()

    if not torch.cuda.is_available():
        print("CUDA not available, exiting.")
        return

    matmul_benchmark(device=args.device, size=args.matmul_size)

    training_like_benchmark(
        device=args.device,
        amp=not args.no_amp,
    )

    h2d_benchmark(device=args.device)

    if not args.skip_disk:
        # on shared FS this may be slow, but that's what we want to see
        disk_benchmark(path=args.disk_path)


if __name__ == "__main__":
    main()
