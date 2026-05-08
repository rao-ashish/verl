# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import inspect
import logging
import os
from contextlib import contextmanager
from typing import Any, Optional

import torch
import torch.distributed as dist
from codetiming import Timer

from verl.utils.device import get_device_id, get_torch_device
from verl.utils.logger import DecoratorLoggerBase


def _get_current_mem_info(unit: str = "GB", precision: int = 2) -> tuple[str]:
    """Get current memory usage.

    Note that CPU device memory info is always 0.

    Args:
        unit (str, optional): The unit of memory measurement. Defaults to "GB".
        precision (int, optional): The number of decimal places to round memory values. Defaults to 2.

    Returns:
        tuple[str]: A tuple containing memory allocated, memory reserved, memory used, and memory total
        in the specified unit.
    """
    assert unit in ["GB", "MB", "KB"]
    device = get_torch_device()
    # torch.cpu.memory_allocated() does not exist
    if device == torch.cpu:
        return "0.00", "0.00", "0.00", "0.00"

    divisor = 1024**3 if unit == "GB" else 1024**2 if unit == "MB" else 1024
    mem_allocated = get_torch_device().memory_allocated()
    mem_reserved = get_torch_device().memory_reserved()
    # use get_torch_device().mem_get_info to profile device memory
    # since vllm's sleep mode works below pytorch
    # see https://github.com/vllm-project/vllm/pull/11743#issuecomment-2754338119
    mem_free, mem_total = get_torch_device().mem_get_info()
    mem_used = mem_total - mem_free
    mem_allocated = f"{mem_allocated / divisor:.{precision}f}"
    mem_reserved = f"{mem_reserved / divisor:.{precision}f}"
    mem_used = f"{mem_used / divisor:.{precision}f}"
    mem_total = f"{mem_total / divisor:.{precision}f}"
    return mem_allocated, mem_reserved, mem_used, mem_total


def log_gpu_memory_usage(
    head: str,
    logger: logging.Logger = None,
    level=logging.DEBUG,
    rank: int = 0,
    synchronize: bool = True,
    per_process: bool = True,
    unit: str = "GB",
    precision: int = 2,
):
    """Log GPU memory usage information.

    Args:
        head (str): A descriptive header for the memory usage log message.
        logger (logging.Logger, optional): Logger instance to use for logging. If None, prints to stdout.
        level: Logging level to use. Defaults to logging.DEBUG.
        rank (int): The rank of the process to log memory for. Defaults to 0.
        synchronize (bool): If True, drain pending GPU work via
            ``torch.<device>.synchronize()`` before sampling memory. This makes
            the reported numbers reflect a quiesced device rather than racing
            with kernels still in flight (cudaMemGetInfo / memory_allocated
            return whatever the allocator and driver currently see, which can
            transiently undercount or overcount when work is queued). Defaults
            to True; set False on hot paths where the extra sync is too costly.
        per_process (bool): If True (default), also emit a per-PID NVML
            breakdown of GPU residency on this rank's device. This is the most
            direct way to attribute bytes that don't show up in PyTorch's
            caching allocator (e.g. vLLM's CuMemAllocator sleep residual,
            NCCL communicators, TE/cuBLAS workspaces, separate Ray actor CUDA
            contexts). NVML attributes bytes to the *process* that called
            ``cuMemAlloc``/``cuMemMap``, so when multiple Ray actors share a
            physical GPU (WorkerDict + vLLMHttpServer + EngineCore + per-TP
            Worker subprocesses) this breakdown tells you which process owns
            the residual. Falls back gracefully to "unavailable" if NVML /
            pynvml is not usable on this device.
        unit (str): Unit for the per-process breakdown. "GB", "MB", or "KB".
            Defaults to "GB". (The base allocator/device line is always
            reported in GB to preserve historical formatting.)
        precision (int): Decimal places for the per-process breakdown.
            Defaults to 2.
    """
    if not ((not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank)):
        return

    if synchronize:
        device = get_torch_device()
        if device != torch.cpu and hasattr(device, "synchronize"):
            device.synchronize()

    mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
    message = (
        f"{head}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, "
        f"device memory used/total (GB): {mem_used}/{mem_total}"
    )

    if per_process:
        rows, free_bytes, total_bytes = _get_per_process_mem_info(unit=unit, precision=precision)
        if rows is None or free_bytes is None or total_bytes is None:
            per_process_message = (
                f"{head} (per-process GPU memory): unavailable (NVML/pynvml not usable on this device)"
            )
        else:
            divisor = 1024**3 if unit == "GB" else 1024**2 if unit == "MB" else 1024
            used_bytes = total_bytes - free_bytes
            header = (
                f"{head} (per-process GPU memory, {unit}): "
                f"device used/total: {used_bytes / divisor:.{precision}f}/{total_bytes / divisor:.{precision}f}, "
                f"n_procs: {len(rows)}"
            )
            if not rows:
                per_process_message = header + " [no compute processes reported by NVML]"
            else:
                self_pid = os.getpid()
                lines = [header]
                for pid, _used_bytes, used_str, name in rows:
                    marker = " <-- self" if pid == self_pid else ""
                    lines.append(f"  pid={pid:<7} mem={used_str:>8} {unit} name={name}{marker}")
                per_process_message = "\n".join(lines)
        message = message + "\n" + per_process_message

    if logger is None:
        print(message)
    else:
        logger.log(msg=message, level=level)


def _get_process_name(pid: int) -> str:
    """Best-effort lookup of a human-readable name for a PID.

    Reads /proc/<pid>/comm (the kernel-set short name, e.g. "ray::WorkerDict"
    when Ray sets the process title) and falls back to the first token of
    /proc/<pid>/cmdline. Returns "?" if neither is readable. Never raises.
    """
    try:
        with open(f"/proc/{pid}/comm") as f:
            comm = f.read().strip()
        if comm:
            return comm
    except OSError:
        pass
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read().split(b"\x00")
        if raw and raw[0]:
            return os.path.basename(raw[0].decode(errors="replace"))
    except OSError:
        pass
    return "?"


def _get_per_process_mem_info(unit: str = "GB", precision: int = 2):
    """Per-process GPU memory usage on the current accelerator, via NVML.

    Returns a list of ``(pid, used_bytes, used_str, name)`` tuples sorted by
    descending memory, plus the device's free/total bytes. NVML reports the
    same physical-GPU view that ``cudaMemGetInfo`` does, so summing
    ``used_bytes`` across rows should approximately equal device "used" from
    :func:`_get_current_mem_info` (modulo bookkeeping NVML doesn't attribute
    to a PID, e.g. CUDA context overhead in some driver versions).

    Returns ``(None, None, None)`` if NVML / pynvml is unavailable or this
    is a non-CUDA device. Never raises.
    """
    assert unit in ["GB", "MB", "KB"]
    device = get_torch_device()
    if device == torch.cpu:
        return None, None, None
    try:
        import pynvml
    except ImportError:
        return None, None, None

    divisor = 1024**3 if unit == "GB" else 1024**2 if unit == "MB" else 1024
    nvml_initialized = False
    try:
        pynvml.nvmlInit()
        nvml_initialized = True
        try:
            device_idx = int(get_device_id())
        except Exception:
            device_idx = 0
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # Try v3 first (richer info on newer drivers); fall back to the
        # version-agnostic API. Both return objects with .pid and .usedGpuMemory.
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses_v3(handle)
        except (AttributeError, pynvml.NVMLError):
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            except pynvml.NVMLError:
                procs = []
        rows = []
        for p in procs:
            used = getattr(p, "usedGpuMemory", None)
            # NVML returns a sentinel (often 2**64-1) when the driver can't
            # attribute memory to the PID (e.g. MIG without per-process
            # accounting). Skip those rather than reporting 16 EiB.
            if used is None or used >= (1 << 63):
                used = 0
            rows.append(
                (
                    int(p.pid),
                    int(used),
                    f"{int(used) / divisor:.{precision}f}",
                    _get_process_name(int(p.pid)),
                )
            )
        rows.sort(key=lambda r: r[1], reverse=True)
        return rows, int(mem_info.free), int(mem_info.total)
    except Exception:
        return None, None, None
    finally:
        if nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


class GPUMemoryLogger(DecoratorLoggerBase):
    """A decorator class to log GPU memory usage.

    Example:
        >>> from verl.utils.profiler.performance import GPUMemoryLogger
        >>> @GPUMemoryLogger(role="actor")
        >>> def update_actor(self, batch):
        ...     # real actor update logics
        ...     return
    """

    def __init__(self, role: str, logger: logging.Logger = None, level=logging.DEBUG, log_only_rank_0: bool = True):
        if dist.is_initialized() and dist.get_world_size() > 1:
            rank = dist.get_rank()
        else:
            rank = 0
        super().__init__(role, logger, level, rank, log_only_rank_0)

    def __call__(self, decorated_function: callable):
        def f(*args, **kwargs):
            return self.log(decorated_function, *args, **kwargs)

        return f

    def log(self, func, *args, **kwargs):
        name = func.__name__
        mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
        message = (
            f"Before {name}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, "
            f"device memory used/total (GB): {mem_used}/{mem_total}"
        )
        self.logging_function(message)

        output = func(*args, **kwargs)

        mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
        message = (
            f"After {name}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, "
            f"device memory used/total (GB): {mem_used}/{mem_total}"
        )

        self.logging_function(message)
        return output


def log_print(ctn: Any):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    frame = inspect.currentframe().f_back
    function_name = frame.f_code.co_name
    line_number = frame.f_lineno
    file_name = frame.f_code.co_filename.split("/")[-1]
    print(f"[{current_time}-{file_name}:{line_number}:{function_name}]: {ctn}")


def _timer(name: str, timing_raw: dict[str, float]):
    """Inner function that handles the core timing logic.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


@contextmanager
def simple_timer(name: str, timing_raw: dict[str, float]):
    """Context manager for basic timing without NVTX markers.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    yield from _timer(name, timing_raw)


@contextmanager
def marked_timer(
    name: str,
    timing_raw: dict[str, float],
    color: str = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
):
    """Context manager for timing with platform markers.

    This utility function measures the execution time of code within its context,
    accumulates the timing information, and adds platform markers for profiling.
    This function is a default implementation when hardware profiler is not available.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.
        color (Optional[str]): Color for the marker. Defaults to None.
        domain (Optional[str]): Domain for the marker. Defaults to None.
        category (Optional[str]): Category for the marker. Defaults to None.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    yield from _timer(name, timing_raw)


def reduce_timing(
    timing_raw: dict[str, float], reduce_op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.AVG
) -> dict[str, float]:
    """Reduce timing information across all processes.

    This function uses distributed communication to gather and sum the timing
    information from all processes in a distributed environment.

    Args:
        timing_raw (Dict[str, float]): Dictionary containing timing information.

    Returns:
        Dict[str, float]: Reduced timing information.
    """
    if not dist.is_initialized():
        return timing_raw

    key_list, timing_list = [], []
    for key in sorted(timing_raw.keys()):
        key_list.append(key)
        timing_list.append(timing_raw[key])
    timing_list = torch.tensor(timing_list, dtype=torch.float32, device=get_device_id())
    torch.distributed.all_reduce(timing_list, op=reduce_op)
    timing_list = [tensor.item() for tensor in timing_list.to("cpu")]
    timing_generate = {key_list[i]: timing_list[i] for i in range(len(key_list))}
    return timing_generate


def topk_reduce_ratio_min_max(timing: float, k: int = 10) -> tuple[float, float, float]:
    """Calculate topk items take-up ratio, and min/max timing across all ranks."""
    if not dist.is_initialized():
        return -1.0, -1.0, -1.0

    world_size = dist.get_world_size()
    timing_tensor = torch.tensor(timing, dtype=torch.float32, device=get_device_id())
    tensor_list = [torch.zeros(1, dtype=torch.float32, device=get_device_id()) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, timing_tensor)
    tensor_stack = torch.stack(tensor_list)
    timing_min = tensor_stack.min().cpu().item()
    timing_max = tensor_stack.max().cpu().item()
    top_k_percentile = torch.quantile(tensor_stack, 1 - k / 100)
    tail_ratio = torch.mean((tensor_stack > top_k_percentile).float()).cpu().item()
    return tail_ratio, timing_min, timing_max


def gather_timing(timing_raw: dict[str, float]) -> dict[str, list[float]]:
    if not dist.is_initialized():
        return {k: [v] for k, v in timing_raw.items()}

    key_list, timing_list = [], []
    for key in sorted(timing_raw.keys()):
        key_list.append(key)
        timing_list.append(timing_raw[key])

    world_size = torch.distributed.get_world_size()

    object_gather_list = [None] * world_size

    torch.distributed.all_gather_object(object_gather_list, timing_list)

    timing_generate = {
        key_list[i]: [timing_list[i] for timing_list in object_gather_list] for i in range(len(key_list))
    }

    return timing_generate
