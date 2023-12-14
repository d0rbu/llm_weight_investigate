import torch.nn as nn
import os
from transformers import LlamaModel


def get_blocks(model: nn.Module) -> list[nn.Module]:
    if isinstance(model, LlamaModel):
        return model.layers
    
    raise ValueError("Unknown model type!")


def get_parameter_names(model: nn.Module) -> list[str]:
    blocks = get_blocks(model)

    assert len(blocks) > 0, "The model should have at least one block to check the parameters of!"
    
    return [name for name, param in blocks[0].named_parameters()]


def get_nested_parameter(name: str, module: nn.Module) -> nn.Module:
    name_hierarchy = name.split(".")

    for name in name_hierarchy:
        module = getattr(module, name)

    return module


def get_parameter_across_blocks(name: str, model: nn.Module) -> list[nn.Module]:
    blocks = get_blocks(model)

    return [get_nested_parameter(name, block) for block in blocks]


def get_parameter_deltas_across_blocks(name: str, model: nn.Module) -> list[nn.Module]:
    parameters = get_parameter_across_blocks(name, model)

    return [
        [parameters[i] - parameters[j] for j in range(i)]
        for i in range(1, len(parameters))
    ]  # (num_blocks - 1, num_blocks - 1, *)


def get_i_parameter_deltas_across_blocks(name: str, model: nn.Module, i: int = 0) -> list[nn.Module]:
    parameters = get_parameter_across_blocks(name, model)

    return [parameters[j] - parameters[i] for j in range(len(parameters)) if j != i]  # (num_blocks - 1, *)


def get_output_path(model_name: str, filename: str, root_dir: os.PathLike | str = "outputs") -> os.PathLike:
    return os.path.join(root_dir, model_name, filename)
