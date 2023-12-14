import torch as th
import os
import matplotlib.pyplot as plt
from matplotlib import colormaps as cmaps
from helpers import get_output_path


def graph_delta_svdvals_from_i(model_name: str, experiment_dir: os.PathLike | str = "outputs", figures_dir: os.PathLike | str = "figures", i: int = 0):
    output_path = get_output_path(model_name, f"weights_delta_svdvals_from_{i}.pt", experiment_dir)

    output = th.load(output_path)
    cmap = cmaps["inferno"]

    for param_name, svdvals in output.items():
        figure_dir = get_output_path(model_name, f"weights_delta_svdvals_from_{i}", figures_dir)
        figure_path = os.path.join(figure_dir, f"{param_name}.png")
        num_blocks = svdvals.shape[0]

        for _j, singular_values in enumerate(svdvals.detach()):
            j = _j if _j < i else _j + 1
            cmap_pos = (((j - i) / num_blocks) + 1) / 2

            plt.plot(singular_values, color=cmap(cmap_pos))
        
        os.makedirs(figure_dir, exist_ok=True)
        plt.savefig(figure_path)
        plt.clf()

        normalized_figure_path = os.path.join(figure_dir, f"normalized_{param_name}.png")

        for _j, singular_values in enumerate(svdvals.detach()):
            j = _j if _j < i else _j + 1
            cmap_pos = (((j - i) / num_blocks) + 1) / 2

            plt.plot(singular_values / singular_values.sum(), color=cmap(cmap_pos))
        
        plt.savefig(normalized_figure_path)
        plt.clf()

        cumulative_figure_path = os.path.join(figure_dir, f"cumulative_{param_name}.png")

        for _j, singular_values in enumerate(svdvals.detach()):
            j = _j if _j < i else _j + 1
            cmap_pos = (((j - i) / num_blocks) + 1) / 2

            plt.plot((singular_values / singular_values.sum()).cumsum(-1), color=cmap(cmap_pos))
        
        plt.savefig(cumulative_figure_path)
        plt.clf()


for i in range(23, 32):
    graph_delta_svdvals_from_i("NousResearch/Llama-2-7b-hf", i=i)
