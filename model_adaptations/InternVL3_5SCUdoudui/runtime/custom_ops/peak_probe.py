"""Worker-side SDAA peak-memory probe for CP1 evidence.

``torch.sdaa`` peak statistics are *process-local*: with the vLLM V1 engine
the model lives in spawned worker subprocesses, so reading
``max_memory_allocated`` from the parent always yields 0 (measured).

This module is mixed into the vLLM worker class through the supported
``worker_extension_cls`` hook and driven from the client with
``LLM.collective_rpc("reset_peak")`` / ``LLM.collective_rpc("read_peak")``,
so the peak is read from inside the worker that actually holds the weights.

Design constraints honoured:
- no environment lookups and no per-forward branching: these methods run
  only when the client explicitly calls them;
- nothing is monkey-patched and no vendor site-packages file is touched;
- the serving scenario is unchanged (same gpu_memory_utilization).

Usage (see scripts/smoke_internvl.py)::

    llm = LLM(..., worker_extension_cls="custom_ops.peak_probe.PeakProbe")
    llm.collective_rpc("reset_peak")
    ... run the forward under test ...
    per_worker = llm.collective_rpc("read_peak")
"""

WORKER_EXTENSION_CLS = "custom_ops.peak_probe.PeakProbe"


class PeakProbe:
    """Mixed into the vLLM worker class via ``worker_extension_cls``."""

    def reset_peak(self):
        """Zero the SDAA peak-memory counters in this worker process."""
        import torch

        torch.sdaa.synchronize()
        for device in range(torch.sdaa.device_count()):
            torch.sdaa.reset_peak_memory_stats(device)
        torch.sdaa.synchronize()
        return {"current_device": int(torch.sdaa.current_device())}

    def read_peak(self):
        """Read per-device allocated/reserved peaks in this worker process."""
        import torch

        torch.sdaa.synchronize()
        per_device = {}
        for device in range(torch.sdaa.device_count()):
            per_device[str(device)] = {
                "max_alloc_gib": round(
                    torch.sdaa.max_memory_allocated(device) / 1024 ** 3, 3),
                "cur_alloc_gib": round(
                    torch.sdaa.memory_allocated(device) / 1024 ** 3, 3),
                "max_reserved_gib": round(
                    torch.sdaa.max_memory_reserved(device) / 1024 ** 3, 3),
            }
        return {
            "current_device": int(torch.sdaa.current_device()),
            "per_device": per_device,
        }
