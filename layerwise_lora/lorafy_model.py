from transformers import PreTrainedModel, PretrainedConfig
from dataclasses import dataclass


class LoRAfyParameterConfig(PretrainedConfig):
    def __init__(
        self,
        LoRAfied_parameter: str,
        base_parameter: str,
        rank: int = -1,
        **kwargs,
    ) -> None:
        self.to_param: str = LoRAfied_parameter
        self.from_param: str = base_parameter
        self.rank: int = rank

        super().__init__(**kwargs)


class LoRAfyConfig(PretrainedConfig):
    def __init__(
        self,
        *LoRAfy_parameter_configs: LoRAfyParameterConfig,
    ) -> None:
        pass


def LoRAfy(model: PreTrainedModel, 
