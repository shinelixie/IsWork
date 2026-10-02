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
./Ascend-cann-nnal_8.5.0_linux-aarch64.run --install --quiet
```

~/.bashrc
添加下面两个行到末尾
source /usr/local/Ascend/ascend-toolkit/set_env.sh

source /usr/local/Ascend/nnal/atb/set_env.sh

```
# 1. 先安装核心 NPU 环境
uv pip install torch==2.8.0 torch-npu==2.8.0 "numpy<2.0.0"
# 2. 再根据你的备份文件强行还原其他依赖，跳过版本检查
uv pip install -r requirements_ascend.txt --no-deps
```

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
#### qwen3.5

先使用安装[requirements_ascend_qwen35](https://github.com/shinelixie/IsWork/blob/main/requirements_ascend_qwen35.txt)里的包,然后使用下面进行安装,注意下面是使用uv进行安装的,原生的pip要将uv去掉

```
# 升级 vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout a75a5b54c7f76bc2e15d3025d6
git fetch origin pull/34521/head:pr-34521
git merge pr-34521
VLLM_TARGET_DEVICE=empty uv pip install -v .

# 升级 vllm-ascend
uv pip uninstall vllm-ascend -y
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout c63b7a11888e9e1caeeff8
git fetch origin pull/6742/head:pr-6742
git merge pr-6742
uv pip install -v .

# 重新安装 transformers
git clone https://github.com/huggingface/transformers.git
cd transformers
git reset --hard fc9137225880a9d03f130634c20f9dbe36a7b8bf
uv pip install .
```

压力测试脚本
```bash
#!/bin/bash
# 加载必要的环境变量
export LD_PRELOAD="/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64/libjemalloc.so:$LD_PRELOAD"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=0

MODEL_PATH="/root/.cache/modelscope/hub/models/Qwen/Qwen3.5-35B-A3B"

# 使用 throughput 模式，并将参数名对齐
vllm bench throughput \
    --model "$MODEL_PATH" \
    --tensor-parallel-size 4 \
    --num-prompts 100 \
    --input-len 200 \
    --output-len 200 \
    --trust-remote-code \
    --additional-config '{"enable_cpu_binding":true, "multistream_overlap_shared_expert": true, "moe_allreduce_overlap_mode": 1}'
```

服务启动脚本
```
#!/bin/bash

# ==========================================================
# 1. 基础环境配置 (编译器与库路径)
# ==========================================================
# Ubuntu 20.04 源里最高只有 gcc-10（gcc-11 不存在）；
# gcc 9.4（系统默认）缺 <span> 等 C++20 头文件，会让 arctic-inference 等包编译失败
export CC=gcc-10
export CXX=g++-10

# 加载 jemalloc 以优化内存分配效率 (针对 CANN 8.5.0)
export LD_PRELOAD="/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64/libjemalloc.so:$LD_PRELOAD"

# ==========================================================
# 2. 昇腾 NPU 推理专项优化参数
# ==========================================================
# 强制使用成熟的 V0 引擎 (避免 V1 引擎在 Triton 算子上的兼容性问题)
export VLLM_USE_V1=0

# 开启 AIV (AI Vector) 引擎深度优化
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024

# 开启任务队列和内存分段，减少显存碎片
export TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

# 离线模式，避免启动时由于联网校验导致的卡顿
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ==========================================================
# 3. 模型路径定义
# ==========================================================
MODEL_PATH="/root/.cache/modelscope/hub/models/Qwen/Qwen3.5-35B-A3B"

# ==========================================================
# 4. 启动 vLLM API 服务
# ==========================================================
# 注意：已去除 --enforce-eager 以启用图模式 (Graph Mode) 提速
vllm serve "$MODEL_PATH" \
    --served-model-name "qwen3.5" \
    --host 0.0.0.0 \
    --port 8010 \
    --tensor-parallel-size 4 \
    --max-model-len 65536 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    --trust-remote-code \
    --async-scheduling \
    --allowed-local-media-path / \
    --mm-processor-cache-gb 0 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --additional-config '{
        "enable_cpu_binding": true, 
        "multistream_overlap_shared_expert": true, 
        "moe_allreduce_overlap_mode": 1
    }'
```
#### qwen3.6

> **环境基线**：Ascend HDK 25.2.3 / CANN 8.5.0（宿主机提供，容器内不可升级）
> **已验证**：910B4 × 7，vLLM 0.18.0 + vllm-ascend 0.18.0rc1 + Qwen3.6-35B-A3B-w8a8（TP=4），
> 实测 `finish_reason=stop`，KV cache 47.05 GiB，模型加载约 4.5 分钟。

##### 0. 版本配套对照

官方 vLLM Ascend 兼容矩阵里 `v0.18.0` 一行要求如下（注意 CANN 写的是 8.5.1，本项目只能用宿主机的 8.5.0）：

| 组件 | 官方要求 | 本项目实际 | 备注 |
|---|---|---|---|
| CANN | 8.5.1 | **8.5.0** | 容器内无法升级，实测可用 |
| torch | 2.9.0 | 2.9.0 | ✓ |
| torch_npu | 2.9.0.post1+gitXXXX | **2.9.0.post1+gitee7ba04** | **按 Python 版本+架构分变体**，见下表 |
| triton-ascend | 3.2.0.dev20260322 | 3.2.0.dev20260322 | 从华为 OBS 装，不是 PyPI 的 3.2.0 |
| 上游 triton | **必须卸载** | 已卸载 | 否则 `TORCH_LIBRARY` 命名空间冲突 |

**关键**：`torch_npu` 预发布版的 wheel 名**随 Python 版本和 CPU 架构变化**，不能照抄矩阵里的 `+git4c901a4`（那是 cp310 的）。完整映射来自 vllm-ascend v0.18.0 的 `Dockerfile`：

| Python | 架构 | wheel 后缀 |
|---|---|---|
| cp310 | aarch64 | `torch_npu-2.9.0.post1+gitec...` → `+git4c901a4` |
| **cp311** | **aarch64** | **`+gitee7ba04`** ← 本项目使用 |
| cp310 | x86_64 | `+gita74051c` |
| cp311 | x86_64 | `+gitdc51c2d` |

下载地址（华为 OBS，实测可直连）：
```
https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/<wheel 文件名>
```

##### 1. 编译器准备（必需）

```bash
# Ubuntu 20.04 源里最高只有 gcc-10（gcc-11 不存在！）
# gcc 9.4（系统默认）缺 <span> 等 C++20 头文件，会导致 arctic-inference 编译失败：
#   fatal error: span: No such file or directory
apt-get install -y gcc-10 g++-10
export CC=gcc-10
export CXX=g++-10
```

验证：
```bash
echo '#include <span>
int main(){return 0;}' > /tmp/t.cc && g++-10 -std=c++20 /tmp/t.cc -o /tmp/t && echo OK
```

##### 2. 基础依赖

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

uv venv vllm-ascend-env --python 3.11
source vllm-ascend-env/bin/activate

# --index-strategy unsafe-best-match 必须加：torch-npu 正式版只在 PyPI，
# 而华为云源排前面且只有 dev 版，默认的 first-index 策略会找不到
uv pip install --no-deps --index-strategy unsafe-best-match \
  -r requirements_qwen36.txt \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

> `requirements_qwen36.txt` 见 [requirements_qwen36](https://github.com/shinelixie/IsWork/blob/main/requirements_qwen36.txt)，
> 会把 torch 升到 2.9.0 / torch-npu 2.9.0 / numpy 2.2.6 / triton-ascend 3.2.0。

##### 3. 安装 vllm

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout bcf2be96120005e9aea171927f85055a6a5c0cf6
VLLM_TARGET_DEVICE=empty uv pip install --no-deps -v .
```

> ⚠️ **不要加 `--no-build-isolation`**。vllm 的 `pyproject.toml` 用 PEP 639 新 license 写法，
> 需要 `setuptools>=77`；而 `requirements_qwen36.txt` 锁的是 69.5.1，加了会报：
> `ValueError: invalid pyproject.toml config: 'project.license'`

> `VLLM_TARGET_DEVICE=empty` 构建出的是**纯 Python wheel**（没有 `vllm/_C*.so`），
> 这是 vllm-ascend 的标准配合方式，**属正常现象**，不是安装失败。

##### 4. 安装 vllm-ascend（需要编译，约 8 分钟）

```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout 99e1ea0fe685e93f53ee5adfe4b41cdd42fb809f

export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0   # set_env.sh 已设置
export MAX_JOBS=48
uv pip install --no-deps -v .
```

编译会产出约 30 个 `.so`，包括 `vllm_ascend_C.cpython-311-aarch64-linux-gnu.so` 和
`_cann_ops_custom/vendors/vllm-ascend/` 下的 CANN 自定义算子。验证：

```bash
python -c "import vllm_ascend; print('OK')"
```

##### 5. 换装官方预发布 torch_npu / triton-ascend，并卸载上游 triton

这一步是照搬 vllm-ascend v0.18.0 的 Dockerfile，**顺序不能改**：

```bash
OBS="https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend"

# (1) 卸载上游 triton —— 否则与 triton-ascend 注册同一个 triton 命名空间
uv pip uninstall triton

# (2) triton-ascend 换 dev 版
uv pip install --no-deps "$OBS/triton_ascend-3.2.0.dev20260322-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"

# (3) torch_npu 换官方预发布版
uv pip install --no-deps "$OBS/torch_npu-2.9.0.post1%2Bgitee7ba04-cp311-cp311-manylinux_2_28_aarch64.whl"
```

> 第 (1) 步不加会报：`RuntimeError: Only a single TORCH_LIBRARY can be used to register the namespace triton`

##### 6. 从源码安装 transformers

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
git reset --hard fc9137225880a9d03f130634c20f9dbe36a7b8bf

# 注意：这一步必须带依赖！transformers 5.2.0.dev0 要求 huggingface-hub>=1.3.0,<2.0
uv pip install .
```

> 若用 `--no-deps` 装会报：`ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'`
> （env 里是给 transformers 4.57.6 用的 huggingface-hub 0.36.2，需要升到 1.33.0）

##### 7. ⚠️ 修复 vLLM 的 rope 参数类型 bug（关键，否则模型起不来）

**这是本项目踩到的最深的坑**：vLLM 把 `ignore_keys_at_rope_validation` 写成了 **list**，
而 transformers 按 **set** 使用，`list | set` 直接抛 TypeError：

```
File "transformers/modeling_rope_utils.py", line 651, in convert_rope_params_to_dict
    ignore_keys_at_rope_validation = ignore_keys_at_rope_validation | {"partial_rotary_factor"}
TypeError: unsupported operand type(s) for |: 'list' and 'set'
```

transformers 自己的模型（`qwen3_vl`、`qwen2_vl`）**全部用 set**，所以是 vLLM 侧写错了。
**两个文件都有这个 bug**，都要改：

| 文件 | 行号 |
|---|---|
| `vllm/transformers_utils/configs/qwen3_5.py` | ~71 |
| `vllm/transformers_utils/configs/qwen3_5_moe.py` | ~78 |

```bash
cd /data/xzh/vllm
patch -p1 < rope-fix.patch        # 补丁见本仓库 rope-fix.patch
```

补丁内容（`[` → `{`、`]` → `}`）：
```diff
-        kwargs["ignore_keys_at_rope_validation"] = [
+        kwargs["ignore_keys_at_rope_validation"] = {
             "mrope_section",
             "mrope_interleaved",
-        ]
+        }
```

> **注意**：vllm 是**非 editable** 安装（代码被复制进 `site-packages`），
> 所以除了改源码树，**还要同步修改** `site-packages/vllm/transformers_utils/configs/` 下的同名文件。
> 或者改完源码树后重新 `uv pip install --no-deps .`。

##### 8. 启动服务

`ascend_llm.bash` 见上面 qwen3.5 的服务启动脚本，qwen3.6 只需改 `MODEL_PATH` 和 `--served-model-name`：

```bash
export CC=gcc-10
export CXX=g++-10
export LD_PRELOAD="/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64/libjemalloc.so:$LD_PRELOAD"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 注意：VLLM_USE_V1 在 vllm 0.18.0 已废弃，会告警 "Unknown vLLM environment variable"，可删

MODEL_PATH="/root/.cache/modelscope/hub/models/Eco-Tech/Qwen3.6-35B-A3B-w8a8"

vllm serve "$MODEL_PATH" \
    --served-model-name "qwen3.6" \
    --host 0.0.0.0 --port 8010 \
    --tensor-parallel-size 4 \
    --max-model-len 65536 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    --trust-remote-code \
    --async-scheduling \
    --allowed-local-media-path / \
    --mm-processor-cache-gb 0 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --additional-config '{
        "enable_cpu_binding": true,
        "multistream_overlap_shared_expert": true,
        "moe_allreduce_overlap_mode": 1
    }'
```

**调用验证**（容器内 8010 端口默认未映射到宿主机，需 SSH 隧道：

```bash
ssh -p 8080 -L 8010:127.0.0.1:8010 root@<宿主机IP>
# 然后本地访问 http://127.0.0.1:8010/v1
```

```bash
curl -s http://127.0.0.1:8010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6","messages":[{"role":"user","content":"你好"}],"max_tokens":100}'
```

##### 踩坑速查表

| 现象 | 原因 | 解决 |
|---|---|---|
| `fatal error: span: No such file or directory` | gcc 9.4 缺 C++20 头文件 | 装 **gcc-10**（`gcc-11` 在 20.04 不存在），设 `CC/CXX` |
| 明明设了 `CC=gcc-10`，编译仍用 `/usr/bin/c++` | **uv 缓存里残留旧 `CMakeCache.txt`**，记录了上次的编译器 | `rm -rf /root/.cache/uv/sdists-v9/pypi/<包名>/` |
| `invalid pyproject.toml config: 'project.license'` | 加了 `--no-build-isolation`，用了 env 里的 setuptools 69.5.1 | **去掉** `--no-build-isolation` |
| `No module named 'vllm._C'` | `VLLM_TARGET_DEVICE=empty` 本就是纯 Python 构建 | **正常，无需处理** |
| `Only a single TORCH_LIBRARY ... namespace triton` | 上游 triton 与 triton-ascend 冲突 | `uv pip uninstall triton` |
| `TypeError: unsupported operand type(s) for \|: 'list' and 'set'` | **vLLM 的 bug**（见步骤 7） | 打 `rope-fix.patch` |
| `cannot import name 'is_offline_mode' from 'huggingface_hub'` | transformers 装时用了 `--no-deps` | 去掉 `--no-deps`，让 hub 升到 1.33.0 |
| `no version of torch-npu==2.9.0 ... first index` | uv 默认只认第一个含该包的索引 | 加 `--index-strategy unsafe-best-match` |
| 端口 8010 从外部连不上 | 容器未发布该端口（只发布了 8080/ttyd） | 用 SSH 隧道，或在 `docker run` 时 `-p` |

