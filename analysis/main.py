import torch as th
import torch.nn as nn
import os
from tqdm import tqdm
from transformers import AutoModel
from functools import partial
from helpers import (
    get_parameter_across_blocks,
    get_parameter_names,
    get_parameter_deltas_across_blocks,
    get_i_parameter_deltas_across_blocks,
    get_output_path
)


def load_model(name: str) -> nn.Module:
    print(f"Loading model {name}...")

    return AutoModel.from_pretrained(
        name, device_map="cpu"
    )


def weights_svdvals(model: nn.Module) -> dict[str, list[th.Tensor]]:
    parameter_names = get_parameter_names(model)

    svdvals = {}

    for name in tqdm(parameter_names):
        parameters = get_parameter_across_blocks(name, model)

        svdvals[name] = th.stack([th.linalg.svdvals(param) for param in tqdm(parameters)], dim=0)

    return svdvals  # name: (num_blocks, num_singular_values)


def weights_delta_svdvals(model: nn.Module) -> dict[str, list[list[th.Tensor]]]:
    parameter_names = get_parameter_names(model)

    svdvals = {}

    for name in tqdm(parameter_names):
        all_parameter_deltas = get_parameter_deltas_across_blocks(name, model)

        ragged_svdval_array = [
            [th.linalg.svdvals(parameter_delta) for parameter_delta in tqdm(parameter_deltas)]
            for parameter_deltas in tqdm(all_parameter_deltas)
        ]  # (num_blocks - 1, num_blocks - 1, num_singular_values)

        svdvals_array = th.zeros((len(all_parameter_deltas), len(all_parameter_deltas), ragged_svdval_array[0][0].shape[0]))

        for i, row in enumerate(ragged_svdval_array):
            for j, current_svdvals in enumerate(row):
                svdvals_array[i + 1, j] = current_svdvals

        svdvals[name] = svdvals_array

    return svdvals  # name: (num_blocks, num_blocks, num_singular_values)


def weights_delta_svdvals_from_i(model: nn.Module, i: int = 0) -> dict[str, list[list[th.Tensor]]]:
    parameter_names = get_parameter_names(model)

    svdvals = {}

    for name in tqdm(parameter_names):
        i_parameter_deltas = get_i_parameter_deltas_across_blocks(name, model, i)

        if len(i_parameter_deltas[0].shape) != 2:
            print(f"parameter is of dimension 1, skipping: {name}")
            continue

        svdvals_array = th.stack(
            [th.linalg.svdvals(parameter_delta.cuda()).cpu() for parameter_delta in tqdm(i_parameter_deltas)],
            dim = 0,
        )  # (num_blocks - 1, num_singular_values)

        svdvals[name] = svdvals_array

    return svdvals  # name: (num_blocks, num_singular_values)


EXPERIMENT_NAMES = {
    "weights_svdvals": weights_svdvals,
    # "weights_delta_svdvals": weights_delta_svdvals,
    
    **{f"weights_delta_svdvals_from_{i}": partial(weights_delta_svdvals_from_i, i=i) for i in range(32)},
}


def main(model_name: str = "NousResearch/Llama-2-7b-hf", experiment: list[str] | str | None = None, experiment_dir: os.PathLike | str = "outputs") -> None:
    model = load_model(model_name)

    if experiment is None:
        experiment = list(EXPERIMENT_NAMES.keys())

    if isinstance(experiment, str):
        experiment = [experiment]

    for experiment_name in experiment:
        print(f"Running experiment {experiment_name}...")
        experiment_fn = EXPERIMENT_NAMES[experiment_name]

        experiment_result = experiment_fn(model)

        print(f"Saving experiment {experiment_name}...")

        experiment_path = get_output_path(model_name, f"{experiment_name}.pt", experiment_dir)
        os.makedirs(os.path.dirname(experiment_path), exist_ok=True)

        th.save(experiment_result, experiment_path)
        del experiment_result


if __name__ == "__main__":
    main(experiment="weights_delta_svdvals_from_{i}")
