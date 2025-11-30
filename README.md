# LLM Fine-tuning Project

本项目旨在探索在有限显存资源（如 Kaggle T4/P100, 16GB）下，对大型语言模型（如 Qwen2-7B）进行高效微调的多种方法。

## 🚀 微调方法概述

本项目涵盖了三种主流的参数高效微调（PEFT）技术，它们的核心思想都是**冻结预训练模型的大部分参数，只训练少量额外参数**，从而大幅降低显存需求。

### 1. LoRA (Low-Rank Adaptation)
*   **原理**：假设模型权重更新矩阵具有“低秩”特性。LoRA 在每个冻结的 Transformer 层旁边增加两个低秩矩阵 $A$ (降维) 和 $B$ (升维)。前向传播时，输出为 $h = W_0x + BAx$。
*   **优点**：不增加推理延迟（训练后可合并权重），显存占用极低。
*   **适用场景**：通用微调，目前最主流的方法。

### 2. QLoRA (Quantized LoRA)
*   **原理**：LoRA 的进化版。它将基础模型量化为 **4-bit (NF4)** 格式以极致压缩显存，同时在 LoRA 的 adapter 中使用 **BFloat16** 进行计算。此外还引入了分页优化器（Paged Optimizer）来防止显存峰值溢出。
*   **优点**：让单张消费级显卡（如 T4, RTX 3060）也能微调 7B/13B 模型。
*   **适用场景**：显存极其受限的环境（如 Kaggle, Colab）。

### 3. AdaLoRA (Adaptive LoRA)
*   **原理**：LoRA 的升级版。LoRA 对所有层分配相同的秩（Rank），而 AdaLoRA 认为不同层的重要性不同。它使用奇异值分解（SVD）在训练过程中**动态分配秩**：对重要层分配更高的秩，对不重要层进行修剪。
*   **优点**：参数分配更合理，通常能用更少的总参数量达到比 LoRA 更好的效果。
*   **关键技术**：需要 `orth_reg_weight` 进行正交正则化，且需要知晓总训练步数来进行秩的调度。

---

## 📄 核心脚本详解: `qwen_adalora_kaggle.py`

这是本项目中最完善、功能最强的微调脚本，专为 **Kaggle T4 (16GB)** 环境设计，实现了 **Qwen2-7B-Instruct** 模型的 **AdaLoRA** 微调。

### 主要功能
1.  **环境适配**：自动检测 GPU，强制使用单卡模式（`CUDA_VISIBLE_DEVICES="0"`）以解决 AdaLoRA 在多卡环境下的兼容性 Bug。
2.  **4-bit 量化**：使用 `BitsAndBytesConfig` 将模型加载为 NF4 格式，将显存占用压缩至约 5.5GB。
3.  **AdaLoRA 配置**：
    *   实现了动态秩调整（Rank Scheduling），初始 Rank=12，目标 Rank=4。
    *   配置了正交正则化（Orthogonality Regularization）以保证 SVD 的有效性。
    *   **关键修复**：显式传递 `total_step`，解决了 AdaLoRA 无法初始化的问题。
4.  **数据处理**：
    *   自动加载本地 IMDB 数据集 (`imdb_samples_2000.csv`)。
    *   **指令格式化**：将数据转换为 `Human: ... Assistant: ...` 的对话格式。
    *   **数据集切分**：自动执行 90% 训练 / 10% 验证切分。
5.  **训练策略优化**：
    *   **Gradient Accumulation**: 设为 8，配合 Batch Size 1，实现等效 Batch Size 8，稳定梯度。
    *   **Evaluation**: 每 50 步验证一次，监控 Loss 和 Next Token Accuracy。
    *   **Checkpointing**: 自动保存验证集 Loss 最低的最佳模型 (`load_best_model_at_end=True`)。

### 如何运行
```bash
python qwen_adalora_kaggle.py
```

### 关键代码片段
```python
# AdaLoRA 动态秩配置示例
peft_config = AdaLoraConfig(
    init_r=12, target_r=4,
    total_step=total_train_steps,  # 必须指定总步数
    tinit=200, tfinal=600,         # 秩更新的时间窗口
    orth_reg_weight=0.5            # 正交正则化权重
)
```

## 📦 依赖安装
请参考 `requirements.txt` 安装所需环境：
```bash
pip install -r requirements.txt
```
