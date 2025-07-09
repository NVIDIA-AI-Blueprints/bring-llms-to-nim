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

Deploy your own Large Language Model (LLM) using NVIDIA NIM - a powerful service that automatically optimizes your model for maximum performance on NVIDIA GPUs.

## What is NIM?

NVIDIA NIM is a containerized solution that:
- **Automatically optimizes** your LLM for the best performance
- **Handles all the complexity** of model deployment and serving
- **Provides an OpenAI-compatible API** for easy integration
- **Supports multiple model formats** without manual conversion

## What You'll Learn

This repository contains three hands-on notebooks showing different deployment approaches:

### 📘 [Notebook 1: Deploy HuggingFace Safetensors](deploy/1_HuggingFace_Safetensors.ipynb)
**Best for:** Quick deployment of popular models from HuggingFace
- Deploy models directly from HuggingFace with a single command
- Download and deploy from local storage for offline use
- Choose between TensorRT-LLM and vLLM backends

### 📗 [Notebook 2: Deploy TensorRT-LLM Checkpoints and Engines](deploy/2_TRTLLM_Checkpoints_Engines.ipynb)
**Best for:** Maximum performance with custom optimization
- Convert HuggingFace models to TensorRT-LLM format
- Compile optimized engines for production deployment
- Fine-tune performance parameters

### 📙 [Notebook 3: Deploy GGUF Checkpoints](deploy/3_GGUF_Checkpoints.ipynb)
**Best for:** Memory-efficient deployment with quantized models
- Deploy pre-quantized GGUF models (4-bit, 8-bit, etc.)
- Reduce GPU memory requirements significantly
- Handle external configuration requirements

> [!IMPORTANT]
> NVIDIA cannot guarantee the security of any models hosted on non-NVIDIA systems such as HuggingFace. Malicious or insecure models can result in serious security risks up to and including full remote code execution. We strongly recommend that before attempting to load it you manually verify the safety of any model not provided by NVIDIA, through such mechanisms as a) ensuring that the model weights are serialized using the safetensors format, b) conducting a manual review of any model or inference code to ensure that it is free of obfuscated or malicious code, and c) validating the signature of the model, if available, to ensure that it comes from a trusted source and has not been modified.

## What You'll Learn
- **What is NIM?** NVIDIA Inference Microservice (NIM) is a containerized solution that automatically optimizes LLM deployment
- **Why use NIM?** Handles model optimization, backend selection, and scaling automatically
- **What you need to know:** Basic Docker commands and Python. No prior LLM deployment experience required.

## Quick Start: Deploy Your First Model in 3 Minutes

### Step 1: Pull the NIM container
```bash
docker pull nvcr.io/nim/nvidia/llm-nim:latest
```

### Step 2: Deploy a small model
```bash
docker run -it --rm --name=my-first-nim \
  --runtime=nvidia --gpus all \
  -p 8000:8000 \
  -e NIM_MODEL_NAME="hf://Qwen/Qwen2.5-0.5B" \
  nvcr.io/nim/nvidia/llm-nim:latest
```

### Step 3: Test it
```bash
curl http://localhost:8000/v1/completions \
  -d '{"model": "Qwen/Qwen2.5-0.5B", "prompt": "Hello world"}'
```

## Overview
This blueprint demonstrates how to deploy almost any Large Language Model using NVIDIA NIM (NVIDIA Inference Microservice). NIM provides a streamlined way to deploy and serve LLMs with optimized performance, handling model analysis, backend selection, and configuration automatically.

Key capabilities:
- Deploy models directly from Hugging Face Hub
- Deploy models from local storage
- Multiple inference backends (TensorRT-LLM, vLLM)
- Customizable deployment parameters
- OpenAI-compatible REST API

#### Software Components
- **NVIDIA NIM LLM Container**: `nvcr.io/nim/nvidia/llm-nim:latest`
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




