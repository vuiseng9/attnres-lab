# pip install --force-reinstall "transformers==5.1.0"

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModel

model_id = "moonshotai/Kimi-K3"

# Downloads only config.json and the small custom Python implementation.
config = AutoConfig.from_pretrained(
    model_id,
    trust_remote_code=True,
)

# Creates every PyTorch module and Parameter on the meta device.
# No checkpoint shards are downloaded or materialized.
with init_empty_weights(include_buffers=True):
    model = AutoModel.from_config(
        config,
        trust_remote_code=True,
    )

model.eval()

print("end.")