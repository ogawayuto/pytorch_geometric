import torch


def ensure_device(device: torch.device):
    r""" Make sure that current cuda kernel corresponds to the assigned device.
  """
    if (device.type == 'cuda' and device.index != torch.cuda.current_device()):
        torch.cuda.set_device(device)
