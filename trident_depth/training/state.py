"""Epoch-boundary training state, without executing optimization steps."""
import random
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_state(generator, epoch, best_rmse, best_epoch):
    numpy = np.random.get_state()
    return dict(epoch=epoch, best_rmse=best_rmse, best_epoch=best_epoch,
                python_rng=random.getstate(), numpy_rng=dict(kind=numpy[0], keys=numpy[1].tolist(),
                    position=numpy[2], has_gauss=numpy[3], cached_gaussian=numpy[4]),
                torch_rng=torch.get_rng_state(), loader_rng=generator.get_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_state(state, generator):
    random.setstate(state["python_rng"])
    numpy = state["numpy_rng"]
    np.random.set_state((numpy["kind"], np.asarray(numpy["keys"], np.uint32), numpy["position"],
                         numpy["has_gauss"], numpy["cached_gaussian"]))
    torch.set_rng_state(state["torch_rng"].cpu())
    generator.set_state(state["loader_rng"].cpu())
    if state["cuda_rng"]:
        if not torch.cuda.is_available() or len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("resuming CUDA RNG state requires the same device count")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
