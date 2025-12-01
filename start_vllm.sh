#!/bin/bash

# ==========================================
# vLLM 启动脚本
# ==========================================

# 确保已经运行了 python merge_model.py 生成了合并后的模型
MODEL_PATH="/kaggle/working/qwen2_spider_merged"

if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 找不到合并后的模型路径 $MODEL_PATH"
    echo "请先运行 python merge_model.py"
    exit 1
fi

echo "正在启动 vLLM 服务器..."
echo "模型路径: $MODEL_PATH"

# 参数说明:
# --tensor-parallel-size 2: 使用 2 张 GPU 进行张量并行 (适合 T4x2 环境)
# --gpu-memory-utilization 0.9: 显存预分配比例
# --trust-remote-code: 允许执行模型自定义代码 (Qwen 需要)
# --dtype float16: 强制使用 float16 (T4 不支持 bfloat16)

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name $MODEL_PATH \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --dtype float16 \
    --port 8000
