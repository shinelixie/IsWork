## LLM
### sft
训练环境： global: uv envirment，相关包在[uv_global_env_pip_freeze](https://github.com/shinelixie/IsWork/blob/main/uv_global_env_pip_freeze.txt) Using Python 3.12.12 environment 

单卡A800使用swift sft lora微调4b模型
```bash

#!/bin/bash
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export SWIFT_PATCH_CONV3D=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
# NPROC_PER_NODE=2 \
# CUDA_VISIBLE_DEVICES=0,1 \

# 显著提升了 batch_size 并开启了 packing 和 deepspeed # --deepspeed zero3_offload \ --gradient_checkpointing true \ --packing true \ --train_type full \ --save_total_limit 3 \
uv run swift sft \
    --resume_from_checkpoint /data/xzh/models/qwen3_4b_tea_agent/v6-20260123-220753/checkpoint-1500 \
    --ignore_data_skip true \
    --model /data/xzh/models/Qwen3-VL-4B-TEA-Extended \
    --train_type lora \
    --cached_dataset /data/xzh/cache/tea_agent_cache/train \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_length 8192 \
    --dataset_num_proc 32 \
    --dataloader_num_workers 16 \
    --attn_impl "flash_attn" \
    --output_dir /data/xzh/models/qwen3_4b_tea_agent \
    --split_dataset_ratio 0.01 \
    --num_train_epochs 3 \
    --per_device_eval_batch_size 1 \
    --padding_free true \
    --learning_rate 2e-5 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --packing true \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --warmup_ratio 0.05 \
```


### eval
评估环境： eval Using Python 3.10.12 environment，[uv_eval_env_pip_freeze](https://github.com/shinelixie/IsWork/blob/main/uv_eval_env_pip_freeze)

启动vllm
```shell
VLLM_USE_MODELSCOPE=True CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /data/xzh/models/qwen3_2b_tea_agent/v1-20260123-173409/checkpoint-1854 \
  --port 8000 \
  --trust-remote-code \
  --max_model_len 32768 \
  --served-model-name qwen3_2b_tea_agent
```
使用evalscope的VLMEvalKit后端配置，开始训练，使用siliconflow api进行评测模型输出结果

export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com 
uv run python eval_math_vista_with_vlmevalkit_backend.py

eval_math_vista_with_vlmevalkit_backend.py
```python
task_cfg_dict = TaskConfig(
    work_dir='outputs',
    eval_backend='VLMEvalKit',
    eval_config={
        "reuse": False,
        'data': ['MathVista_MINI'],
        # 'limit': 20,
        'mode': 'all',
        'model': [ 
            {'api_base': 'http://localhost:8000/v1/chat/completions',
            'key': 'EMPTY',
            'name': 'CustomAPIModel',
            'temperature': 0.0,
            'type': 'qwen3_2b_tea_agent',
            'img_size': -1,
            'video_llm': False,
            'max_tokens': 30000,}
            ],
        'nproc': 2,
        'judge': 'exact_matching',
        'OPENAI_API_KEY' : "",
        'OPENAI_API_BASE' : "https://api.siliconflow.cn/v1/chat/completions",
        'LOCAL_LLM' : 'deepseek-ai/DeepSeek-V3.2',
        },

)
```

使用evalscope的VLMEvalKit后端配置，从已有/中断的评测结果继续评测
```python
task_cfg_dict = TaskConfig(
    work_dir='outputs',
    use_cache="outputs/20260124_213449",
    eval_backend='VLMEvalKit',
    eval_config={
        "reuse": True,
        'data': ['MathVista_MINI'],
        # 'limit': 20,
        'mode': 'all',
        'model': [ 
            {'api_base': 'http://localhost:8000/v1/chat/completions',
            'key': 'EMPTY',
            'name': 'CustomAPIModel',
            'temperature': 0.0,
            'type': 'qwen3_2b_tea_agent',
            'img_size': -1,
            'video_llm': False,
            'max_tokens': 30000,}
            ],
        'nproc': 2,
        'judge': 'exact_matching',
        'OPENAI_API_KEY' : "",
        'OPENAI_API_BASE' : "https://api.siliconflow.cn/v1/chat/completions",
        'LOCAL_LLM' : 'deepseek-ai/DeepSeek-V3.2',
        },

)
```

使用swift的eval，无法配置裁判模型

```bash
CONDA_DEV_PATH="/home/xzh/miniconda3/envs/swift_eval_env"
export CPATH=$CONDA_DEV_PATH/include/python3.10:$CPATH
export C_INCLUDE_PATH=$CONDA_DEV_PATH/include/python3.10:$C_INCLUDE_PATH
export LIBRARY_PATH=$CONDA_DEV_PATH/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_DEV_PATH/lib:$LD_LIBRARY_PATH

# VLLM_USE_MODELSCOPE=True CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-VL-3B-Instruct --port 8000 --trust-remote-code --max_model_len 4096 --served-model-name Qwen2.5-VL-3B-Instruct

# 3. 执行评测
CUDA_VISIBLE_DEVICES=0 \
swift eval \
    --model /data/xzh/models/qwen3_2b_tea_agent/v1-20260123-173409/checkpoint-1854 \
    --model_type qwen3_vl \
    --eval_backend VLMEvalKit \
    --eval_dataset MathVista_MINI \
    --infer_backend vllm \
    --eval_dataset_args '{
        "MathVista_MINI": {
            "local_path": "/data/xzh/datasets/MathVista"
        }
    }'
```
