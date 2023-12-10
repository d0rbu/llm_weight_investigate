import torch as th
import torch.nn as nn
from transformers import AutoModel
from helpers import get_parameter_across_blocks, get_parameter_names


def load_model(name: str) -> nn.Module:
    return AutoModel.from_pretrained(
        name, device_map="auto"
    )


def weights_svd(model: nn.Module) -> dict[str, list[th.Tensor]]:
    # TODO: for each parameter type, perform svd on all of the blocks in the model and get the singular values
    pass


def weights_delta_svd(model: nn.Module) -> dict[str, list[list[th.Tensor]]]:
    # TODO: same as above except do it for all the deltas of weights from each other
    pass


def main(model_name: str = "NousResearch/Llama-2-70b-hf", experiment: list[str] | str | None = None) -> None:
    pass


if __name__ == "__main__":
    main()
