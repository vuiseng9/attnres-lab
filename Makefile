ngpu ?= $(shell nvidia-smi -L | wc -l)
torchrun_intra = torchrun --standalone --nproc-per-node
dbg ?= 0
xargs +=

# install torch on your own to align to your system
setup:
	uv pip install -r requirements.txt

__train_100m__:
	DBG_ATTACH=$(dbg) $(torchrun_intra) $(ngpu) train.py \
	  --mode $(MODE) \
	  --hidden_size 512 \
	  --num_layers 12 \
	  --num_heads 8 \
	  --num_kv_heads 4 \
	  --intermediate_size 1536 \
	  --seq_len 2048 \
	  --steps 20000 \
	  --batch_size 1 \
	  --grad_accum 8

00_train_baseline: 
	$(MAKE) __train_100m__ MODE=baseline

0b_train_block:
	$(MAKE) __train_100m__ MODE=block

0f_train_full:
	$(MAKE) __train_100m__ MODE=full


__eval_model__:
	python eval.py --model_path $(model_id) --mode $(MODE)

90_eval_100m_baseline: 
	$(MAKE) __eval_model__ MODE=baseline model_id=Ethangou/attention-residuals-100M-baseline

9f_eval_100m_full:
	$(MAKE) __eval_model__ MODE=full model_id=Ethangou/attention-residuals-100M-full

9b_eval_100m_block:
	$(MAKE) __eval_model__ MODE=block model_id=Ethangou/attention-residuals-100M-block
