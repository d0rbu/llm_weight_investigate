import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from lorafy_model import LoRAfiedModel
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from main import LORAFIED_MODEL_DIR


if __name__ == "__main__":
    name = "NousResearch/Llama-2-7b-hf"
    device = "cuda:0"
    tasks = ["hellaswag"]
    path = os.path.join(LORAFIED_MODEL_DIR, name)

    base_model = AutoModelForCausalLM.from_pretrained(name)
    model = LoRAfiedModel.from_pretrained(path, model=base_model)
    tokenizer = AutoTokenizer.from_pretrained(path)

    lm = HFLM(pretrained=model, tokenizer=tokenizer, device=device)

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        batch_size="auto",
        device=device,
    )
    
    import pdb; pdb.set_trace()
