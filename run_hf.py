import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
import torch

config = LlamaConfig(use_cache=True)



tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
llama_model = AutoModelForCausalLM(config=config, pretrained_model_name_or_path="meta-llama/Llama-2-7b-hf")

prompt = "The meaning of life is"

input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = llama_model.generate(input_ids, max_length=100, num_return_sequences=5, temperature=1)