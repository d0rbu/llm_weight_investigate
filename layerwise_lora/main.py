import os
import itertools
from transformers import AutoModelForCausalLM, AutoTokenizer
from lorafy_model import LoRAfyParameterConfig, LoRAfyConfig, LoRAfiedModel
from lm_eval import evaluator

LORAFIED_MODEL_DIR = "lorafied_models"


def lorafied_llama_2_7b():
    MODEL_NAME = "NousResearch/Llama-2-7b-hf"
    WEIGHT_TYPES = ("self_attn.q_proj", "self_attn.k_proj")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    parameter_configs = [
        *(LoRAfyParameterConfig(
            to_param=f"model.layers.{i}.{weight_type}",
            from_param=f"model.layers.0.{weight_type}",
            rank=1024,
            initialize=True,
        ) for i, weight_type in itertools.product(range(1, 32), WEIGHT_TYPES))
    ]
    config = LoRAfyConfig(model, *parameter_configs, do_sample=True)

    return LoRAfiedModel(config, model), tokenizer, MODEL_NAME

if __name__ == "__main__":
    model, tokenizer, name = lorafied_llama_2_7b()
    path = os.path.join(LORAFIED_MODEL_DIR, name)

    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
