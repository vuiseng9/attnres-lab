from transformers import AutoTokenizer, set_seed
import torch
from modeling_attnres import Qwen3AttnResForCausalLM

set_seed(42)

model_id = "Ethangou/attention-residuals-100M-full"
prompt = "I love the Avengers, "
max_new_tokens = 32

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = Qwen3AttnResForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    attn_implementation="eager",
).to(device).eval()

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))