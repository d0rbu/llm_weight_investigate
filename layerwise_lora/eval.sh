lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks winogrande,openbookqa,mmlu,hellaswag,lambada,piqa,coqa,gsm8k,boolq,arc_easy,arc_challenge \
    --device cuda \
    --batch_size auto