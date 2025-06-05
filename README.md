<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<h2><img align="center" src="https://github.com/user-attachments/assets/cbe0d62f-c856-4e0b-b3ee-6184b7c4d96f">NVIDIA AI Blueprint: Bring Your LLM to NIM</h2>

Deploy any Large Language Model (LLM) using NVIDIA NIM with optimized performance and flexibility across multiple inference backends.


> [!IMPORTANT]
> NVIDIA cannot guarantee the security of any models hosted on non-NVIDIA systems such as HuggingFace. Malicious or insecure models can result in serious security risks up to and including full remote code execution. We strongly recommend that before attempting to load it you manually verify the safety of any model not provided by NVIDIA, through such mechanisms as a) ensuring that the model weights are serialized using the safetensors format, b) conducting a manual review of any model or inference code to ensure that it is free of obfuscated or malicious code, and c) validating the signature of the model, if available, to ensure that it comes from a trusted source and has not been modified.

### Quickstart
1. **Set up prerequisites**: Obtain NGC API Key and Hugging Face Token
2. **Deploy a model**: Run the following command to deploy Codestral-22B:
   ```bash
   docker run -it --rm \
     --name=Universal-LLM-NIM \
     --runtime=nvidia \
     --gpus all \
     --shm-size=16GB \
     -e HF_TOKEN=<your_hf_token> \
     -e NIM_MODEL_NAME="hf://mistralai/Codestral-22B-v0.1" \
     -e NIM_SERVED_MODEL_NAME="mistralai/Codestral-22B-v0.1" \
     -v "$HOME/.cache/nim:/opt/nim/.cache" \
     -u $(id -u) \
     -p 8000:8000 \
     -d \
     nvcr.io/nvidian/nim-llm-dev/universal-nim:1.11.0.rc4
   ```
3. **Test the deployment**:
   ```bash
   curl -X POST "http://localhost:8000/v1/completions" \
     -H "Content-Type: application/json" \
     -d '{"model": "mistralai/Codestral-22B-v0.1", "prompt": "Write a Python function to compute fibonacci", "max_tokens": 250}'
   ```

### Overview
This blueprint demonstrates how to deploy almost any Large Language Model using NVIDIA NIM (NVIDIA Inference Microservice). NIM provides a streamlined way to deploy and serve LLMs with optimized performance, handling model analysis, backend selection, and configuration automatically.

Key capabilities:
- Deploy models directly from Hugging Face Hub
- Deploy models from local storage
- Multiple inference backends (TensorRT-LLM, vLLM)
- Customizable deployment parameters
- OpenAI-compatible REST API

#### Software Components
- **NVIDIA NIM Universal LLM Container**: `nvcr.io/nvidian/nim-llm-dev/universal-nim:1.11.0.rc4`
- **Supported Inference Backends**:
  - TensorRT-LLM (high performance)
  - vLLM (versatile deployment)
- **Container Runtime**: Docker with NVIDIA Container Runtime
- **Model Sources**: Hugging Face Hub, Local filesystem

#### Hardware Requirements
- **GPU**: NVIDIA GPU with compute capability 7.0+ (V100, T4, A10, A100, H100, etc.)
- **GPU Memory**: Varies by model size (minimum 8GB for small models, 40GB+ for large models)
- **System Memory**: 16GB+ RAM recommended
- **Storage**: Sufficient disk space for model weights and cache
- **NVIDIA Driver**: Version 535+ required

#### API Definition
The deployed NIM service provides OpenAI-compatible REST API endpoints:

- **Health Check**: `GET /v1/health/ready`
- **Completions**: `POST /v1/completions`
- **Chat Completions**: `POST /v1/chat/completions`
- **Models**: `GET /v1/models`

Example completion request:
```json
{
  "model": "model-name",
  "prompt": "Your prompt here",
  "max_tokens": 250,
  "temperature": 0.7
}
```

### Deployment

#### Prerequisites
- **NVIDIA Developer License**: Access to NVIDIA Container Registry
- **NGC API Key**: Required for pulling NIM containers ([Generate here](https://docs.nvidia.com/ngc/gpu-cloud/ngc-user-guide/index.html#generating-api-key))
- **Hugging Face Token**: Required for downloading models from Hugging Face Hub
- **Docker**: With NVIDIA Container Runtime installed
- **NVIDIA Driver**: Version 535 or higher

**Setup Steps**:
1. Install NVIDIA drivers and Docker with NVIDIA runtime
2. Log into NGC registry:
   ```bash
   echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
   ```
3. Set environment variables for API keys

#### Deployment Options

**Option 1: Deploy from Hugging Face Hub**
```bash
docker run -it --rm \
  --name=Universal-LLM-NIM \
  --runtime=nvidia \
  --gpus all \
  --shm-size=16GB \
  -e HF_TOKEN=$HF_TOKEN \
  -e NIM_MODEL_NAME="hf://model-repo/model-name" \
  -e NIM_SERVED_MODEL_NAME="model-name" \
  -v "$HOME/.cache/nim:/opt/nim/.cache" \
  -u $(id -u) \
  -p 8000:8000 \
  -d \
  nvcr.io/nvidian/nim-llm-dev/universal-nim:1.11.0.rc4
```

**Option 2: Deploy from Local Model**
```bash
docker run -it --rm \
  --name=Universal-LLM-NIM \
  --runtime=nvidia \
  --gpus all \
  --shm-size=16GB \
  -e NIM_MODEL_NAME="/opt/models/model-name" \
  -e NIM_SERVED_MODEL_NAME="model-name" \
  -v "/path/to/local/model:/opt/models/model-name" \
  -v "$HOME/.cache/nim:/opt/nim/.cache" \
  -u $(id -u) \
  -p 8000:8000 \
  -d \
  nvcr.io/nvidian/nim-llm-dev/universal-nim:1.11.0.rc4
```

### Customization

#### Backend Selection
Specify inference backend using `NIM_MODEL_PROFILE`:
- `tensorrt_llm`: High-performance TensorRT-LLM backend
- `vllm`: Versatile vLLM backend

#### Performance Tuning Parameters
- `NIM_TENSOR_PARALLEL_SIZE`: Number of GPUs for tensor parallelism
- `NIM_MAX_BATCH_SIZE`: Maximum batch size for inference
- `NIM_MAX_INPUT_LENGTH`: Maximum input sequence length (default: 2048)
- `NIM_MAX_OUTPUT_LENGTH`: Maximum output sequence length (default: 512)

#### Example Custom Deployment
```bash
docker run -it --rm \
  --name=Universal-LLM-NIM \
  --runtime=nvidia \
  --gpus all \
  --shm-size=16GB \
  -e NIM_MODEL_PROFILE="tensorrt_llm" \
  -e NIM_TENSOR_PARALLEL_SIZE=2 \
  -e NIM_MAX_BATCH_SIZE=16 \
  -e NIM_MAX_INPUT_LENGTH=2048 \
  -e NIM_MAX_OUTPUT_LENGTH=512 \
  # ... other parameters
```

For detailed examples and code samples, see the [deployment notebook](deploy/1_Deploy_NIM.ipynb).



