import os
from lorafy_model import LoRAfyParameterConfig, LoRAfyConfig, LoRAfiedModel
from transformers import AutoModelForCausalLM

LORAFIED_MODEL_DIR = "lorafied_models"


def test_lorafy_config():
    parameter_config = LoRAfyParameterConfig(to_param="test1", from_param="test2", rank=5)
    config = LoRAfyConfig(parameter_config, do_sample=True)

    config.save_pretrained(LORAFIED_MODEL_DIR)
    loaded_config = LoRAfyConfig.from_pretrained(LORAFIED_MODEL_DIR)

    assert config.to_dict() == loaded_config.to_dict(), "Saving and loading config failed!"


def test_lorafy_llama_2_7b():
    MODEL_NAME = "NousResearch/Llama-2-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    parameter_configs = [
        LoRAfyParameterConfig(
            to_param=f"model.layers.{i}.self_attn.q_proj",
            from_param="model.layers.0.self_attn.q_proj",
            rank=1000,
        ) for i in range(1, 32)
    ]
    config = LoRAfyConfig(*parameter_configs, model=model, do_sample=True)

    lorafied_model = LoRAfiedModel(config, model)
    
    import pdb; pdb.set_trace()
    lorafied_model.save_pretrained(os.path.join(LORAFIED_MODEL_DIR, MODEL_NAME))


test_lorafy_config()
