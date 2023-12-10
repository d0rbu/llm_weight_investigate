import torch.nn as nn
from transformers import LlamaModel


def get_blocks(model: nn.Module) -> list[nn.Module]:
    if isinstance(model, LlamaModel):
        return model.layers
    
    raise ValueError("Unknown model type!")


def get_parameter_names(model: nn.Module) -> list[str]:
    blocks = get_blocks(model)

    assert len(blocks) > 0, "The model should have at least one block to check the parameters of!"
    
    return [name for name, param in blocks[0].named_parameters()]


def get_parameter_across_blocks(name: str, model: nn.Module) -> list[nn.Module]:
    blocks = get_blocks(model)

    return [getattr(block, name) for block in blocks]
