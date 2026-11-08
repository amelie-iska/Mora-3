"""Synchronized bounded profiling; timings are observations, not speedup claims."""
import copy
import time
from contextlib import contextmanager
import torch


@contextmanager
def preserved_training_state(trainer):
    """Restore native checkpoint, gradients, module modes and all torch RNGs.

    The caller owns exclusive access to the trainer during this diagnostic.
    External data/files and arbitrary Python callback side effects are not restored.
    """
    state = copy.deepcopy(trainer.state_dict())
    parameters = tuple(trainer.model.parameters())
    gradients = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
    modes = [(m, m.training) for m in trainer.model.modules()]
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        trainer.load_state_dict(state)
        for p, grad in zip(parameters, gradients):
            p.grad = grad
        for module, training in modes:
            module.training = training
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def measure_phase(name, function, *, device):
    """Return callback result and synchronized wall time with CUDA allocation peak.

    CUDA peaks are absolute allocated bytes for this phase, including live state;
    resetting allocator peak counters is an explicit profiling side effect.
    CPU memory is deliberately unavailable rather than estimated from CUDA.
    """
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = function()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return result, dict(phase=name, device=str(device), seconds=elapsed,
        peak_allocated_bytes=(torch.cuda.max_memory_allocated(device)
                              if device.type == 'cuda' else None))
