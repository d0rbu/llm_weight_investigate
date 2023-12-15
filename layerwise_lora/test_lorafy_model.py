import os
from lorafy_model import LoRAfyParameterConfig, LoRAfyConfig, LoRAfiedModel
from transformers import AutoModelForCausalLM

TEST_DIR = "test_output"


def test_lorafy_config():
    parameter_config = LoRAfyParameterConfig(to_param="test1", from_param="test2", rank=5)
    config = LoRAfyConfig(parameter_config, do_sample=True)

    config.save_pretrained(TEST_DIR)
    loaded_config = LoRAfyConfig.from_pretrained(TEST_DIR)

    assert config.to_dict() == loaded_config.to_dict(), "Saving and loading config failed!"
