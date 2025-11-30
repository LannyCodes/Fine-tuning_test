# ==========================================
# Kaggle Qwen2-7B AdaLoRA 微调脚本
# ==========================================
# 参照 AdaLoRA_gpu.py 的逻辑，将其应用于 Qwen2-7B 模型
# 结合了 Kaggle 的 GPU 环境限制 (T4/P100)
# ==========================================

import os
# 强制只使用第一张显卡，避免 AdaLoRA 在多卡环境下因张量跨设备导致的 RuntimeError
# AdaLoRA 的正则化计算目前在 device_map="auto" 分布式加载时存在已知兼容性问题
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import os
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    AdaLoraConfig,  # 使用 AdaLoRA
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)

# --- 配置部分 ---
MODEL_ID = "Qwen/Qwen2-7B-Instruct"
OUTPUT_DIR = "/kaggle/working/qwen2_adalora_output"

def train_qwen_adalora():
    print(f"正在检查环境...")
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 GPU")
    
    # Kaggle T4/P100 兼容性设置
    compute_dtype = torch.float16
    print(f"使用计算精度: {compute_dtype}")

    # 1. 量化配置 (4-bit QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # 2. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 加载模型
    print("正在加载模型...")
    # 在多卡环境下，AdaLoRA 的正则化计算（orth_reg_weight > 0）可能会因为张量不在同一设备导致 RuntimeError
    # Kaggle 环境虽然通常是 T4 x2，但为了稳妥，建议强制 device_map="auto" 
    # 并且在 Trainer 中禁用数据并行（如果使用 Accelerate 会自动处理，但 AdaLoRA 比较特殊）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 4. 准备数据
    print("正在加载 IMDB 数据集...")
    # 使用相对路径，方便上传到 GitHub 后其他人也能直接运行
    data_file = os.path.join(os.path.dirname(__file__), "imdb_samples_2000.csv")
    if not os.path.exists(data_file):
        # 兼容 Kaggle 环境或不同工作目录的情况
        data_file = "imdb_samples_2000.csv"
        
    dataset = load_dataset("csv", data_files=data_file, split="train")
    
    # 只使用前 1000 条数据
    dataset = dataset.select(range(1000))

    def process_func(example):
        # 构建指令微调格式
        # IMDB 数据集包含 text 和 label (positive/negative)
        prompt = f"Human: 请判断下面这段电影评论的情感倾向（positive 或 negative）。\n评论内容：{example['text']}\nAssistant: {example['label']}"
        
        inputs = tokenizer(
            prompt + tokenizer.eos_token, 
            truncation=True, 
            max_length=512,
            padding="max_length"
        )
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs

    tokenized_ds = dataset.map(process_func, remove_columns=dataset.column_names)

    # 划分训练集和验证集 (90% 训练, 10% 验证)
    # 标准做法：
    # 1. 训练集 (Train): 用于梯度更新，模型从中学习。
    # 2. 验证集 (Eval/Val): 用于训练过程中的评估，观察 Loss 是否下降，防止过拟合。
    # 3. 测试集 (Test): 训练结束后才用，完全不参与训练，用于最终效果打分 (本脚本中可以把验证集兼作测试集，或单独再留一份)
    split_dataset = tokenized_ds.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(eval_dataset)}")

    # 计算总训练步数 (用于 AdaLoRA 调度)
    per_device_batch_size = 1
    gradient_accumulation_steps = 8 # 等效 batch_size = 8
    num_epochs = 3  # 数据量增加后，增加轮数以充分训练
    
    # 注意：这里使用 train_dataset 的长度来计算
    num_update_steps_per_epoch = len(train_dataset) // (per_device_batch_size * gradient_accumulation_steps)
    total_train_steps = num_update_steps_per_epoch * num_epochs
    print(f"预计总训练步数: {total_train_steps}")

    # 5. AdaLoRA 配置
    # 参照 AdaLoRA_gpu.py 的逻辑配置参数
    # 注意：AdaLoRA 相比 LoRA 会增加一些计算开销用于 SVD 更新
    peft_config = AdaLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        
        # 动态秩配置
        init_r=12,           # 初始秩 (稍高一点，让它有空间修剪)
        target_r=4,          # 目标平均秩 (希望最终压缩到的大小)
        beta1=0.85,          # SVD 平滑系数
        beta2=0.85,
        
        # 关键：必须提供总训练步数
        total_step=total_train_steps,
        
        # 关键：更新调度 (确保 tfinal < total_step)
        tinit=int(total_train_steps * 0.2),         # 20% 步数时开始
        tfinal=int(total_train_steps * 0.6),        # 60% 步数时结束
        deltaT=max(1, int(total_train_steps * 0.05)), # 更新频率
        
        lora_alpha=32,
        lora_dropout=0.1,
        
        # Qwen2 的目标模块
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        
        # 必须指定 orth_reg_weight 以进行正交正则化
        # 我们已经通过 CUDA_VISIBLE_DEVICES="0" 解决了多卡冲突问题，
        # 现在可以安全地恢复此参数，保证 AdaLoRA 的效果。
        orth_reg_weight=0.5,
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 定义评估指标计算函数
    import numpy as np
    
    def preprocess_logits_for_metrics(logits, labels):
        """
        预处理 logits 以节省显存。
        logits 的形状通常是 [batch_size, seq_len, vocab_size]，非常大。
        我们不需要完整的 logits 来计算准确率，只需要 argmax。
        """
        if isinstance(logits, tuple):
            # 根据模型不同，logits 可能是 tuple (logits, past_key_values)
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_metrics(eval_preds):
        """
        计算 Next Token Accuracy (预测下一个 token 的准确率)
        preds: 来自 preprocess_logits_for_metrics 的 argmax 结果
        labels: 真实标签
        """
        preds, labels = eval_preds
        
        # 展平
        preds = preds.reshape(-1)
        labels = labels.reshape(-1)
        
        # 过滤掉 padding token (通常 label 为 -100)
        mask = labels != -100
        preds = preds[mask]
        labels = labels[mask]
        
        # 计算准确率
        accuracy = (preds == labels).mean()
        return {"accuracy": accuracy}

    # 6. 训练参数
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=1,   # 验证集批次大小
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=10,               # 每 10 步记录一次日志
        num_train_epochs=num_epochs,
        learning_rate=1e-4,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        
        # 添加验证策略
        eval_strategy="steps",    # 新版 Transformers 使用 eval_strategy
        eval_steps=50,                  # 每 50 步验证一次 (约半个 epoch)
        
        # 关键：根据验证集指标保存最佳模型
        save_strategy="steps",          # 必须开启保存，load_best_model_at_end 才能生效
        save_steps=50,                  # 与 eval_steps 保持一致
        load_best_model_at_end=True,    # 训练结束后自动加载验证集效果最好的模型
        metric_for_best_model="eval_loss", # 以 loss 为准 (越小越好)
        # 或者使用 metric_for_best_model="accuracy", greater_is_better=True
        
        report_to="none",
        # 必须为 AdaLoRA 留出足够的步数进行 rank 更新，否则会报错或不生效
        # 如果 total_train_steps 太小，建议增加 epoch
    )

    # 7. 开始训练
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,    # 使用划分后的训练集
        eval_dataset=eval_dataset,      # 使用划分后的验证集
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        compute_metrics=compute_metrics, # 添加指标计算
        preprocess_logits_for_metrics=preprocess_logits_for_metrics, # 关键：节省显存
    )

    print("开始 AdaLoRA 训练...")
    trainer.train()
    
    # 8. 训练结束后，进行最终评估
    # 虽然我们在训练过程中已经进行了验证，但在 load_best_model_at_end=True 的情况下，
    # 再次运行 evaluate() 可以确认当前加载的“最佳模型”的具体指标。
    print("\n正在对最佳模型进行最终评估...")
    final_metrics = trainer.evaluate()
    print("最终验证集指标:", final_metrics)
    
    # 保存
    print(f"保存 AdaLoRA 适配器到 {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 打印最终的秩分布 (AdaLoRA 特性)
    print("\n--- AdaLoRA 最终秩分布 ---")
    # 这是一个查看各层最终保留了多少 Rank 的辅助函数
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "lora" in name:
            # 注意：peft 实现细节可能会变，这里尝试获取 rank
            try:
                # 某些版本的 peft 在 Linear 层上直接有 r 属性，或者在 active_adapter 中
                pass 
            except:
                pass
    print("训练完成！")

if __name__ == "__main__":
    train_qwen_adalora()
