import numpy as np
import torch as th
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from typing import Iterable, Any, Hashable, TypeVar

H = TypeVar("H", bound=Hashable)


class LoRAfyParameterConfig(PretrainedConfig):
    def __init__(
        self,
        LoRAfied_parameter: str,
        base_parameter: str,
        rank: int = -1,
        **kwargs: dict[str, Any],
    ) -> None:
        self.to_param: str = LoRAfied_parameter
        self.from_param: str = base_parameter
        self.rank: int = rank

        super().__init__(**kwargs)


def get_duplicate_elements(iter: Iterable[H]) -> set[H]:
    seen = set()
    dupes = set()
    
    for element in iter:
        if element in seen:
            dupes.add(element)
        else:
            seen.add(element)
    
    return dupes


class LoRAfyConfig(PretrainedConfig):
    def __init__(
        self,
        model: PreTrainedModel,
        *LoRAfy_parameter_configs: list[LoRAfyParameterConfig],
        default_rank: int = -1,
        **kwargs: dict[str, Any],
    ) -> None:
        to_params = [param_config.to_param for param_config in LoRAfy_parameter_configs]
        duplicate_to_params = get_duplicate_elements(to_params)

        if len(duplicate_to_params) > 0:
            raise ValueError(f"At least one parameter is being mapped to by multiple configs! " \
                             f"The following parameter(s) show up multiple times as targets to " \
                             f"LoRAfy: {duplicate_to_params}")
        
        ranks = np.array([param_config.rank for param_config in LoRAfy_parameter_configs])
        if default_rank < 0:
            undefined_rank_configs_mask = ranks < 0
            if undefined_rank_configs_mask.any():
                raise ValueError(f"A rank must be specified for each LoRAfied parameter, but no " \
                                 f"default rank was specified and at least one parameter config " \
                                 f"did not specify a rank: {ranks[undefined_rank_configs_mask]}")
        
        from_params = [param_config.from_param for param_config in LoRAfy_parameter_configs]
        all_params = set(to_params).union(set(from_params))
        model_state_dict_keys = model.state_dict().keys()
        for param in all_params:
            if param not in model_state_dict_keys:
                raise ValueError(f"Parameter {param} is referenced in the LoRAfy parameter " \
                                 f"configs but it was not found in the model state_dict. Are " \
                                 f"you sure you have the right model loaded?")

        self.model: PreTrainedModel = model
        self.param_configs: Iterable[LoRAfyParameterConfig] = LoRAfy_parameter_configs
        self.default_rank: int = default_rank

        super().__init__(**kwargs)


class LoRA(nn.Module):
    def __init__(self, base: nn.Linear, down_proj: nn.Linear, up_proj: nn.Linear):
        super().__init__()

        self.A = base
        self.Q_T = down_proj
        self.P = up_proj
    
    def forward(self, x: th.Tensor):
        return self.A(x) + self.P(self.Q_T(x))


class LoRAfiedModel(PreTrainedModel):
    config_class = LoRAfyConfig

    def __init__(self, config: LoRAfyConfig) -> None:
        super().__init__(config)

        self.model = config.model

        # Replace each LoRAfied parameter with its respective A + PQ*
        for param_config in config.param_configs:
            rank = param_config.rank if param_config.rank >= 0 else config.default_rank
            self.LoRAfy_parameter(self.model, param_config.from_param, param_config.to_param, rank)

    def get_nested_parameter(self, name: str) -> nn.Module:
        if name == "":
            return self.model

        name_hierarchy = name.split(".")
        module = self.model

        for name in name_hierarchy:
            module = getattr(module, name)

        return module
    
    def LoRAfy_parameter(self, from_param: str, to_param: str, rank: int) -> None:
        to_param_parent = ".".join(to_param.split(".")[:-1])
        from_layer = self.get_nested_parameter(from_param)
        to_layer = self.get_nested_parameter(to_param)
        
        weight_delta = to_layer.weight - from_layer.weight
        U, S, Vh = th.linalg.svd(weight_delta)
        # TODO: finish
        
    def forward(self, x: Any):
        return self.model.forward(x)
