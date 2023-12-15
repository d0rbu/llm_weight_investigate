import numpy as np
import torch as th
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, AutoModel
from typing import Iterable, Any, Hashable, TypeVar, TypedDict

H = TypeVar("H", bound=Hashable)


class LoRAfyParameterConfig(TypedDict):
    to_param: str
    from_param: str
    rank: int | float | None
    initialize: bool | None


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
    model_type = "lorafied"

    def __init__(
        self,
        model: PreTrainedModel | None = None,
        *LoRAfy_parameter_configs: Iterable[LoRAfyParameterConfig],
        default_rank: int | float | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        for parameter_config in LoRAfy_parameter_configs:  # Remove trailing .weight
            for param_type in ("to_param", "from_param"):
                param_hierarchy = parameter_config[param_type].split(".")
                if param_hierarchy[-1] == "weight":
                    param_hierarchy = param_hierarchy[:-1]
                
                parameter_config[param_type] = ".".join(param_hierarchy)

        to_params = [param_config["to_param"] for param_config in LoRAfy_parameter_configs]
        from_params = [param_config["from_param"] for param_config in LoRAfy_parameter_configs]
        all_params = set(to_params).union(set(from_params))

        if None in all_params:
            raise ValueError(f"All base and LoRAfied parameters must be specified in parameter " \
                             f"configs! Found at least one {None}!")

        duplicate_to_params = get_duplicate_elements(to_params)

        if len(duplicate_to_params) > 0:
            raise ValueError(f"At least one parameter is being mapped to by multiple configs! " \
                             f"The following parameter(s) show up multiple times as targets to " \
                             f"LoRAfy: {duplicate_to_params}")

        ranks = np.array([param_config["rank"] for param_config in LoRAfy_parameter_configs])
        if default_rank is None or default_rank < 0:
            error = ValueError(f"A rank must be specified for each LoRAfied parameter, but no " \
                                 f"default rank was specified and at least one parameter config " \
                                 f"did not specify a rank.")

            if None in ranks:
                raise error

            negative_rank_configs_mask = (ranks[ranks != np.array(None)]) < 0
            if negative_rank_configs_mask.any():
                raise error

        if model is not None:  # Check that the parameter configs make sense
            model_state_dict_keys = model.state_dict().keys()
            for param in all_params:
                if f"{param}.weight" not in model_state_dict_keys:
                    raise ValueError(f"Parameter {param} is referenced in the LoRAfy parameter " \
                                    f"configs but {param}.weight was not found in the model " \
                                    f"state_dict. Are you sure you have the right model loaded?")

        self.param_configs: list[LoRAfyParameterConfig] = list(LoRAfy_parameter_configs)
        self.default_rank: int | float = -1 if default_rank is None else default_rank

        if model is not None:
            AutoConfig.register(self.model_type, LoRAfyConfig)
            AutoModel.register(LoRAfyConfig, LoRAfiedModel)

        super().__init__(**kwargs)


class LoRA(nn.Module):
    def __init__(self, base: nn.Linear, up_proj: nn.Linear, down_proj: nn.Linear):
        super().__init__()

        self.A: nn.Linear = base
        self.P: nn.Linear = up_proj
        self.Qh: nn.Linear = down_proj
    
    def forward(self, x: th.Tensor):
        return self.A(x) + self.P(self.Qh(x))


class LoRAfiedModel(PreTrainedModel):
    config_class = LoRAfyConfig

    def __init__(self, config: LoRAfyConfig, model: PreTrainedModel) -> None:
        super().__init__(config)

        self.model = model

        # Replace each LoRAfied parameter with its respective A + PQ*
        for param_config in config.param_configs:
            rank = param_config["rank"] if param_config["rank"] and param_config["rank"] >= 0 \
                else config.default_rank
            calculate_pq: bool = bool(param_config["initialize"])
            param_config["initialize"] = False  # If we already initialized, don't do it again
            self.LoRAfy_parameter(
                param_config["from_param"],
                param_config["to_param"],
                rank,
                calculate_pq,
            )

    def get_nested_parameter(self, name: str) -> nn.Module:
        if name == "":
            return self.model

        name_hierarchy = name.split(".")
        module = self.model

        for name in name_hierarchy:
            module = getattr(module, name)

        return module

    def LoRAfy_parameter(
            self,
            from_param: str,
            to_param: str,
            rank: int | float,
            initialize: bool = True
        ) -> None:
        print(f"LoRAfying {to_param} from {from_param}")
        to_param_hierarchy = to_param.split(".")
        to_param_name = to_param_hierarchy.pop()
        to_layer_parent_name = ".".join(to_param_hierarchy)
        to_layer_parent = self.get_nested_parameter(to_layer_parent_name)
        from_layer = self.get_nested_parameter(from_param)
        to_layer = self.get_nested_parameter(to_param)

        if to_layer.weight.shape != from_layer.weight.shape:
            raise ValueError(f"{to_param} is being LoRAfied from base parameter {from_layer}" \
                             f", but they have different shapes! Base has shape " \
                             f"{from_layer.weight.shape} and LoRAfied weight has shape " \
                             f"{to_layer.weight.shape}")

        if initialize:
            print(f"Approximating {to_param} from {from_param}...")

            weight_delta = to_layer.weight - from_layer.weight
            U, S, Vh = th.linalg.svd(weight_delta, full_matrices=False)
            V = th.conj(Vh).transpose(0, 1)

            if isinstance(rank, float):
                rank: int = int(rank * S.shape[0])

            U_truncated = U[:, :rank]
            S_truncated = S[:rank]
            S_truncated.sqrt_()
            S_truncated = th.diag_embed(S_truncated)
            V_truncated = V[:, :rank]

            # P = US^{1/2}, Q = VS^{1/2}
            P = U_truncated @ S_truncated
            Q = V_truncated @ S_truncated
            Qh = th.conj(Q).transpose(0, 1)

            up_proj_layer = nn.Linear(P.shape[1], P.shape[0])
            down_proj_layer = nn.Linear(*Q.shape)

            with th.no_grad():
                up_proj_layer.weight.copy_(P)
                down_proj_layer.weight.copy_(Qh)
        else:
            up_proj_layer = nn.Linear(rank, to_layer.weight.shape[0])
            down_proj_layer = nn.Linear(to_layer.weight.shape[1], rank)

        new_layer = LoRA(from_layer, up_proj_layer, down_proj_layer)

        setattr(to_layer_parent, to_param_name, new_layer)
        del to_layer

    def forward(self, x: Any):
        return self.model.forward(x)
