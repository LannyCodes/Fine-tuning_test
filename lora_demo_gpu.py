# 1. 设置环境
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像
os.environ["WANDB_DISABLED"] = "true"  # 禁用WandB日志

# 2. 安装必要库
#!pip install -U transformers torch peft datasets accelerate

# 3. 导入库
from transformers import (
    DistilBertTokenizer,  # 使用DistilBERT专用分词器
    DistilBertForSequenceClassification,  # 使用DistilBERT专用模型类
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch
import numpy as np

# 4. 加载模型和分词器
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 5. 加载数据集
dataset = load_dataset("imdb")

# 6. 数据预处理函数
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding=True,
        truncation=True,
        max_length=128
    )

# 应用分词器并重命名标签列
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# 7. 配置LoRA（关键修正：使用正确的目标模块名称）
lora_config = LoraConfig(
    r=4,  # 更低的秩以加速训练
    lora_alpha=8,
    target_modules=["ffn.lin1", "ffn.lin2"],  # DistilBERT的正确目标模块
    # 或者使用注意力层: ["attention.q_lin", "attention.k_lin", "attention.v_lin"]
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"
)

# 8. 创建LoRA模型
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()  # 打印可训练参数

# 9. 创建数据收集器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 10. 训练配置
training_args = TrainingArguments(
    output_dir="./results",
    run_name="lora_imdb_demo",
    per_device_train_batch_size=8,
    num_train_epochs=10,
    learning_rate=1e-3,
    eval_strategy="no",        # 这里使用eval_strategy，并设置为"no"表示不进行评估
    save_strategy="no",
    logging_steps=10,
    report_to="none",
    fp16=True,  # 启用混合精度加速训练
)

# 11. 创建Trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].select(range(100)),
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 12. 开始训练
print("开始训练...")
trainer.train()
print("训练完成！")

# 13. 快速测试函数
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = peft_model(**inputs)
        logits = outputs.logits
    prob = torch.softmax(logits, dim=1)[0]
    return {"negative": prob[0].item(), "positive": prob[1].item()}

# 14. 测试预测
print("\n测试结果：")
print("正面评论:", predict_sentiment("This movie was absolutely wonderful!"))
print("负面评论:", predict_sentiment("Terrible experience, waste of time"))