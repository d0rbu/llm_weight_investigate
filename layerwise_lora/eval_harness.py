import os
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from lorafy_model import LoRAfiedModel
from lm_eval import evaluator
from lm_eval.tasks import initialize_tasks
from lm_eval.models.huggingface import HFLM
from main import LORAFIED_MODEL_DIR


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


if __name__ == "__main__":
    verbosity = "INFO"
    initialize_tasks(verbosity)

    name = "meta-llama/Llama-2-7b-hf"
    # tasks = ["winogrande", "lambada", "piqa", "coqa", "hellaswag", "mmlu"]
    # tasks = ["winogrande"]
    # tasks = ["lambada", "piqa", "coqa"]
    # tasks = ["mmlu"]
    # tasks = ["boolq"]
    # tasks = ["hellaswag"]
    # tasks = ["arc_easy", "arc_challenge"]
    # tasks = ["openbookqa"]
    # tasks = ["commonsenseqa"]
    tasks = ["gsm8k"]
    path = os.path.join(LORAFIED_MODEL_DIR, name)

    base_model = AutoModelForCausalLM.from_pretrained(name)
    model = LoRAfiedModel.from_pretrained(path, model=base_model, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(path)

    lm = HFLM(pretrained=model, tokenizer=tokenizer)

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        batch_size="auto",
    )

    with open("gsm8k_results.json", "w") as f:
        json.dump(results, f, indent=2, default=_handle_non_serializable)
