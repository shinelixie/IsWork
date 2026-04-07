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

### ascend 
Ascend HDK = 25.2.3
uv venv vllm-ascend-env --python 3.11
下载 NNAL 8.5.0 (昇腾社区版链接)

```
wget --header="Referer: https://www.hiascend.com/" https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.0/Ascend-cann-nnal_8.5.0_linux-aarch64.run
```
执行安装
```
chmod +x Ascend-cann-nnal_8.5.0_linux-aarch64.run
```
```
./Ascend-cann-nnal_8.5.0_linux-aarch64.run --install --quiet
```

~/.bashrc
添加下面两个行到末尾
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

[requirements_ascend](https://github.com/shinelixie/IsWork/blob/main/requirements_ascend.txt) 是python所需要安装的包

测试代码
```python
from vllm import LLM, SamplingParams

# 设置模型路径 (指向你刚才下载成功的目录)
model_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"

# 定义测试提示词
prompts = [
    "你好，请介绍一下你自己。",
    "The future of AI on Ascend NPU is",
]

# 设置采样参数
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)

# 初始化 LLM 引擎 (注意：vllm-ascend 会自动识别 NPU)
# 如果是多卡 A2/A3，可以加上 tensor_parallel_size=8
llm = LLM(model=model_path, trust_remote_code=True)

# 生成输出
outputs = llm.generate(prompts, sampling_params)

# 打印结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nResponse: {generated_text!r}\n")
```
在线推理
```bash
# 1. 硬件环境变量
export ASCEND_RT_VISIBLE_DEVICES=0  # 0.6B模型单卡0即可，1TB内存完全溢出
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1

# 2. 路径变量 (确保 uv 环境优先)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 3. 启动服务
# 注意：删除了 --quantization ascend (因为0.6B通常非量化)
# 注意：调整了 TP 为 1
uv run vllm serve /root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
  --served-model-name qwen3-0.6b \
  --trust-remote-code \
  --async-scheduling \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --max-num-batched-tokens 40960 \
  --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
  --port 8113 \
  --gpu-memory-utilization 0.8
```
