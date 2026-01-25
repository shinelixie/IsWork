## LLM
### sft
评估环境： eval

启动vllm
```bash
VLLM_USE_MODELSCOPE=True CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /data/xzh/models/qwen3_2b_tea_agent/v1-20260123-173409/checkpoint-1854 \
  --port 8000 \
  --trust-remote-code \
  --max_model_len 32768 \
  --served-model-name qwen3_2b_tea_agent
```

使用evalscope的VLMEvalKit后端配置，从已有/中断的评测结果继续评测
```
from evalscope import TaskConfig
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['LOCAL_RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
# 强制让某些依赖库不要乱猜并行策略
from evalscope import TaskConfig
from evalscope.constants import  JudgeStrategy

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

from evalscope.run import run_task
from evalscope.summarizer import Summarizer

def run_eval():
    # 选项 1: python 字典
    task_cfg = task_cfg_dict

    # 选项 2: yaml 配置文件
    # task_cfg = 'eval_openai_api.yaml'

    run_task(task_cfg=task_cfg)

    print('>> Start to get the report with summarizer ...')
    report_list = Summarizer.get_report_from_cfg(task_cfg)
    print(f'\n>> The report list: {report_list}')

run_eval()
```


训练环境： global: uv envirment
2026/1/21: datasets num_rows: 7659
pip packages:
Using Python 3.12.12 environment at: global
Package                       Version     Editable project location
----------------------------- ----------- -------------------------
absl-py                       2.3.1
accelerate                    1.12.0
addict                        2.4.0
aiofiles                      24.1.0
aiohappyeyeballs              2.6.1
aiohttp                       3.13.3
aiosignal                     1.4.0
aliyun-python-sdk-core        2.16.0
aliyun-python-sdk-kms         2.16.5
anls                          0.0.2
annotated-doc                 0.0.4
annotated-types               0.7.0
antlr4-python3-runtime        4.7.2
anyio                         4.12.1
apex                          0.1
attrdict                      2.0.1
attrs                         25.4.0
audioread                     3.1.0
av                            15.1.0
beautifulsoup4                4.14.3
binpacking                    1.5.2
black                         25.12.0
blis                          1.3.3
brotli                        1.2.0
capture-metric                0.1.13
catalogue                     2.0.10
cbor                          1.0.0
certifi                       2026.1.4
cffi                          2.0.0
cfgv                          3.5.0
chardet                       5.2.0
charset-normalizer            3.4.4
click                         8.3.1
cloudpathlib                  0.23.0
colorama                      0.4.6
confection                    0.1.5
contourpy                     1.3.3
cpm-kernels                   1.0.11
crcmod                        1.7
cryptography                  46.0.3
cycler                        0.12.1
cymem                         2.0.13
dacite                        1.9.2
dataclasses-json              0.6.7
dataproperty                  1.1.0
datasets                      4.4.2
decorator                     5.2.1
decord                        0.6.0
deepspeed                     0.18.4
dill                          0.3.7
distlib                       0.4.0
distro                        1.9.0
dotenv                        0.9.9
duckduckgo-search             8.1.1
editdistance                  0.8.1
einops                        0.8.1
et-xmlfile                    2.0.0
evaluate                      0.4.6
factualscenegraph             0.6.1
fastapi                       0.128.0
ffmpy                         1.0.0
filelock                      3.20.3
flagembedding                 1.3.5
flash-attn                    2.8.3
fonttools                     4.61.1
frozenlist                    1.8.0
fsspec                        2023.10.0
ftfy                          6.3.1
future                        1.0.0
gitdb                         4.0.12
gitpython                     3.1.46
google-ai-generativelanguage  0.6.15
google-api-core               2.29.0
google-api-python-client      2.188.0
google-auth                   2.47.0
google-auth-httplib2          0.3.0
google-generativeai           0.8.6
googleapis-common-protos      1.72.0
gradio                        5.50.0
gradio-client                 1.14.0
greenlet                      3.3.0
groovy                        0.1.2
grpcio                        1.76.0
grpcio-status                 1.71.2
h11                           0.16.0
hf-transfer                   0.1.9
hf-xet                        1.2.0
hjson                         3.1.0
httpcore                      1.0.9
httplib2                      0.31.1
httpx                         0.28.1
httpx-sse                     0.4.0
huggingface-hub               0.36.0
identify                      2.6.16
idna                          3.11
ijson                         3.4.0.post0
importlib-metadata            8.7.1
inscriptis                    2.6.0
ir-datasets                   0.5.11
isort                         7.0.0
jieba                         0.42.1
jinja2                        3.1.6
jiter                         0.12.0
jmespath                      0.10.0
joblib                        1.5.3
json-repair                   0.55.0
jsonlines                     4.0.0
jsonpatch                     1.33
jsonpointer                   3.0.0
kiwisolver                    1.4.9
langchain                     1.2.3
langchain-classic             1.0.1
langchain-community           0.4.1
langchain-core                1.2.7
langchain-text-splitters      1.1.0
langgraph                     1.0.6
langgraph-checkpoint          4.0.0
langgraph-prebuilt            1.0.6
langgraph-sdk                 0.3.3
langsmith                     0.6.2
latex2sympy2                  1.9.1
lazy-loader                   0.4
levenshtein                   0.27.3
librosa                       0.11.0
llvmlite                      0.46.0
lmms-eval                     0.5.0       /data/xzh/lmms-eval-main
loguru                        0.7.3
lxml                          6.0.2
lz4                           4.4.5
markdown                      3.10
markdown-it-py                4.0.0
markupsafe                    3.0.3
marshmallow                   3.26.2
matplotlib                    3.10.8
mbstrdecoder                  1.1.4
mdurl                         0.1.2
megatron-core                 0.15.3
ml-dtypes                     0.5.4
modelscope                    1.33.0
more-itertools                10.8.0
mpmath                        1.3.0
ms-swift                      3.12.1
msgpack                       1.1.2
multidict                     6.7.0
multiprocess                  0.70.15
murmurhash                    1.0.15
mypy-extensions               1.1.0
networkx                      3.6.1
ninja                         1.13.0
nltk                          3.9.2
nodeenv                       1.10.0
numba                         0.63.1
numexpr                       2.14.1
numpy                         1.26.4
nvidia-cublas-cu12            12.8.4.1
nvidia-cuda-cupti-cu12        12.8.90
nvidia-cuda-nvrtc-cu12        12.8.93
nvidia-cuda-runtime-cu12      12.8.90
nvidia-cudnn-cu12             9.10.2.21
nvidia-cufft-cu12             11.3.3.83
nvidia-cufile-cu12            1.13.1.3
nvidia-curand-cu12            10.3.9.90
nvidia-cusolver-cu12          11.7.3.90
nvidia-cusparse-cu12          12.5.8.93
nvidia-cusparselt-cu12        0.7.1
nvidia-nccl-cu12              2.27.5
nvidia-nvjitlink-cu12         12.8.93
nvidia-nvshmem-cu12           3.3.20
nvidia-nvtx-cu12              12.8.90
omegaconf                     2.3.0
onnx                          1.20.1
onnx-ir                       0.1.14
onnxscript                    0.5.7
openai                        2.15.0
opencv-python-headless        4.11.0.86
openpyxl                      3.1.5
orjson                        3.11.5
ormsgpack                     1.12.1
oss2                          2.19.1
packaging                     25.0
pandas                        2.3.3
pathspec                      1.0.3
pathvalidate                  3.3.1
peft                          0.18.1
pillow                        12.1.0
platformdirs                  4.5.1
playwright                    1.57.0
pooch                         1.8.2
portalocker                   3.2.0
pre-commit                    4.5.1
preshed                       3.0.12
primp                         0.15.0
propcache                     0.4.1
proto-plus                    1.27.0
protobuf                      5.29.5
psutil                        7.2.1
py-cpuinfo                    9.0.0
pyarrow                       22.0.0
pyasn1                        0.6.1
pyasn1-modules                0.4.2
pybind11                      3.0.1
pycocoevalcap                 1.2
pycocotools                   2.0.11
pycparser                     2.23
pycryptodome                  3.23.0
pydantic                      2.12.5
pydantic-core                 2.41.5
pydantic-settings             2.12.0
pydub                         0.25.1
pyee                          13.0.0
pygments                      2.19.2
pyparsing                     3.3.1
pytablewriter                 1.2.1
python-dateutil               2.9.0.post0
python-dotenv                 1.2.1
python-multipart              0.0.21
pytokens                      0.3.0
pytz                          2025.2
pywsd                         1.2.5
pyyaml                        6.0.3
qwen-vl-utils                 0.0.14
rapidfuzz                     3.14.3
regex                         2025.11.3
reka-api                      3.2.0
requests                      2.32.5
requests-toolbelt             1.0.0
rfc3986                       1.5.0
rich                          14.2.0
rouge                         1.0.1
rsa                           4.9.1
ruff                          0.14.11
sacrebleu                     2.6.0
safehttpx                     0.1.7
safetensors                   0.7.0
scikit-learn                  1.8.0
scipy                         1.17.0
semantic-version              2.10.0
sentence-transformers         5.2.0
sentencepiece                 0.2.1
sentry-sdk                    2.49.0
setuptools                    80.9.0
shellingham                   1.5.4
simplejson                    3.20.2
six                           1.17.0
smart-open                    7.5.0
smmap                         5.0.2
sniffio                       1.3.1
sortedcontainers              2.4.0
soundfile                     0.13.1
soupsieve                     2.8.1
soxr                          1.0.0
spacy                         3.8.11
spacy-legacy                  3.0.12
spacy-loggers                 1.0.5
sqlalchemy                    2.0.45
sqlitedict                    2.1.0
srsly                         2.5.2
starlette                     0.50.0
sympy                         1.14.0
tabledata                     1.3.4
tabulate                      0.9.0
tcolorpy                      0.1.7
tenacity                      8.3.0
tensorboard                   2.20.0
tensorboard-data-server       0.7.2
thinc                         8.3.10
threadpoolctl                 3.6.0
tiktoken                      0.12.0
timm                          1.0.24
tokenizers                    0.22.2
tomlkit                       0.13.3
torch                         2.9.1
torchcodec                    0.9.1
torchvision                   0.24.1
tqdm                          4.67.1
tqdm-multiprocess             0.0.11
transformer-engine            2.11.0
transformer-engine-cu12       2.11.0
transformers                  4.57.5
transformers-stream-generator 0.0.5
trec-car-tools                2.6
triton                        3.5.1
trl                           0.24.0
typepy                        1.3.4
typer                         0.21.1
typer-slim                    0.21.1
typing-extensions             4.15.0
typing-inspect                0.9.0
typing-inspection             0.4.2
tzdata                        2025.3
unlzw3                        0.2.3
uritemplate                   4.2.0
urllib3                       2.6.3
uuid-utils                    0.13.0
uvicorn                       0.40.0
virtualenv                    20.36.1
wandb                         0.24.0
warc3-wet                     0.2.5
warc3-wet-clueweb09           0.2.5
wasabi                        1.1.3
wcwidth                       0.2.14
weasel                        0.4.3
websockets                    15.0.1
werkzeug                      3.1.5
wn                            0.0.23
wrapt                         2.0.1
xxhash                        3.6.0
yarl                          1.22.0
yt-dlp                        2025.12.8
zhconv                        1.4.3
zipp                          3.23.0
zlib-state                    0.1.10
zss                           1.2.0
zstandard                     0.25.0
