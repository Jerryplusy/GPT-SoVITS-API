# GPT-SoVITS TTS 部署指南

本文档介绍如何使用 Poetry 部署和运行 GPT-SoVITS TTS 服务。

## 环境要求

- Python 3.11
- Poetry (包管理器)
- FFmpeg (音频处理)

## 安装步骤

### 1. 安装 Poetry

```bash
# macOS
brew install poetry

# Linux
curl -sSL https://install.python-poetry.org | python3 -

# Windows
pip install poetry
```

### 2. 安装 FFmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# CentOS/RHEL
sudo yum install ffmpeg
```

### 3. 安装项目依赖

```bash
cd /path/to/GPT-SoVITS
poetry install
```

### 4. 下载预训练模型

从 HuggingFace 或 ModelScope 下载预训练模型：

```bash
# 创建模型目录
mkdir -p GPT_SoVITS/pretrained_models

# 下载模型文件并放置到对应目录：
# - GPT_SoVITS/pretrained_models/s1v3.ckpt
# - GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth
# - GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/
# - GPT_SoVITS/pretrained_models/fast_langdetect/
# - GPT_SoVITS/pretrained_models/chinese-hubert-base/
```

推荐从以下地址下载：
- [HuggingFace](https://huggingface.co/)
- [ModelScope](https://modelscope.cn/)

## 配置说明

编辑 `config.yaml` 文件：

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  api_key: "your-secret-api-key-here"  # 修改为你自己的密钥

tts:
  # 设备选择: mps (Apple Silicon) / cuda (NVIDIA GPU) / cpu
  device: "mps"

  # 半精度推理 (可提速，仅 GPU 支持)
  is_half: true

  # 模型版本: v1 / v2 / v3 / v4 / v2Pro / v2ProPlus
  version: "v4"

  # 模型路径 (根据实际下载位置调整)
  t2s_weights_path: "GPT_SoVITS/pretrained_models/s1v3.ckpt"
  vits_weights_path: "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth"
  bert_base_path: "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
  cnhuhbert_base_path: "GPT_SoVITS/pretrained_models/chinese-hubert-base"

  # 推理参数
  top_k: 15
  top_p: 1.0
  temperature: 1.0
  batch_size: 1
  batch_threshold: 0.75
  split_bucket: true
  speed_factor: 1.0
  fragment_interval: 0.3
  seed: 42
  parallel_infer: true
  repetition_penalty: 1.35

  # v3/v4 vocoder 采样步数 (8-32，越高越慢但越好)
  sample_steps: 12

  # 音频超采样 (需要额外模型)
  super_sampling: false
```

### 设备配置推荐

| 设备 | device | is_half | sample_steps |
|------|--------|---------|--------------|
| Apple M1/M2/M3/M4 | `mps` | true | 12 |
| NVIDIA GPU (CUDA) | `cuda` | true | 12 |
| CPU | `cpu` | false | 16-24 |

## 使用方法

### 准备参考音频

在 `input/` 目录下放置参考音频和文本：

```
input/
├── ref.mp3       # 参考音频 (3-20秒)
├── ref_text.txt  # 参考文本
└── output.wav     # 输出音频 (自动生成)
```

### 方式一：命令行 (CLI)

```bash
# 交互式输入
poetry run python quick_tts.py

# 直接传入文本
poetry run python quick_tts.py "你好世界"

# 指定配置文件
poetry run python quick_tts.py "你好世界" --config /path/to/config.yaml
```

### 方式二：API 服务

#### 启动服务

```bash
# 默认配置 (0.0.0.0:8000)
poetry run python tts_api.py

# 自定义端口
poetry run python tts_api.py --port 8080

# 开发模式 (自动重载)
poetry run python tts_api.py --reload
```

#### API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/tts` | POST | 生成音频 (JSON 返回 Base64) |
| `/tts/wav` | GET | 生成音频 (直接返回 WAV 文件) |

#### 调用示例

**1. 健康检查**

```bash
curl http://localhost:8000/health
```

响应：
```json
{"status": "healthy"}
```

**2. POST /tts (JSON 返回)**

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key-here" \
  -d '{
    "text": "你好世界",
    "ref_audio_path": "input/ref.mp3",
    "prompt_text": "参考文本内容"
  }'
```

响应：
```json
{
  "audio_base64": "UklGRi...",
  "sample_rate": 32000,
  "text": "你好世界"
}
```

**3. GET /tts/wav (文件返回)**

```bash
# URL 编码参数
curl "http://localhost:8000/tts/wav?text=你好&ref_audio_path=input/ref.mp3&prompt_text=参考文本" \
  -H "X-API-Key: your-secret-api-key-here" \
  -o output.wav
```

**4. Python 调用示例**

```python
import requests
import base64

url = "http://localhost:8000/tts"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "your-secret-api-key-here"
}
data = {
    "text": "你好世界",
    "ref_audio_path": "input/ref.mp3",
    "prompt_text": "参考文本内容"
}

response = requests.post(url, json=data, headers=headers)
result = response.json()

# 解码音频
audio_bytes = base64.b64decode(result["audio_base64"])
with open("output.wav", "wb") as f:
    f.write(audio_bytes)

print(f"Sample rate: {result['sample_rate']} Hz")
```

## 性能优化

### Apple Silicon (M1/M2/M3/M4)

```yaml
tts:
  device: "mps"
  is_half: false  # MPS 建议保持 FP32
  sample_steps: 12
```

### NVIDIA GPU

```yaml
tts:
  device: "cuda"
  is_half: true
  sample_steps: 12
```

### CPU 优化

```yaml
tts:
  device: "cpu"
  is_half: false
  sample_steps: 8  # 降低以提速
  parallel_infer: false  # CPU 上建议关闭并行
```

## 常见问题

### 1. 模型文件路径错误

确保模型文件路径正确，检查文件是否存在：
```bash
ls -la GPT_SoVITS/pretrained_models/
```

### 2. MPS 不可用

检查 PyTorch 是否支持 MPS：
```bash
poetry run python -c "import torch; print(torch.backends.mps.is_available())"
```

### 3. 显存不足

- 降低 `batch_size`
- 关闭 `is_half: false`
- 降低 `sample_steps`

### 4. 音频长度问题

参考音频时长需在 3-20 秒范围内。

## 目录结构

```
GPT-SoVITS/
├── config.yaml          # 配置文件
├── quick_tts.py         # CLI 工具
├── tts_api.py           # API 服务
├── input/               # 输入输出目录
│   ├── ref.mp3
│   ├── ref_text.txt
│   └── output.wav
└── GPT_SoVITS/
    ├── TTS_infer_pack/  # TTS 核心
    ├── pretrained_models/  # 预训练模型
    └── ...
```

## 许可证

本项目基于 GPT-SoVITS 原项目许可证。
