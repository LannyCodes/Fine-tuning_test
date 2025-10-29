# 1. 安装库（保持不变）
# !pip uninstall -y transformers peft
# !pip install transformers==4.38.0 peft==0.7.0 torch datasets accelerate

# 2. 导入库
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from peft import AdaLoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score

# 3. 设置环境
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

# 4. 加载模型和分词器
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 打印所有线性层（帮助确定 target_modules）
print("模型中的所有线性层:")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(name)

# 5. 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
model = model.to(device)


# 6. 加载数据集（增加样本量）
dataset = load_dataset("imdb")
# 训练集用5000样本，测试集用1000样本（避免过拟合）
train_dataset = dataset["train"].shuffle(seed=42).select(range(5000))  # 增加到5000
eval_dataset = dataset["test"].shuffle(seed=42).select(range(1000))   # 增加到1000

# 7. 数据预处理
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",  # 固定长度，避免动态padding的干扰
        truncation=True,
        max_length=128
    )

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
# 重命名标签列
tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_eval = tokenized_eval.rename_column("label", "labels")

# 只保留需要的列（加速训练），避免移除不存在的列
columns_to_remove = ["text"]  # 只移除text列，保留其他可能存在的列
tokenized_train = tokenized_train.remove_columns(columns_to_remove)
tokenized_eval = tokenized_eval.remove_columns(columns_to_remove)

tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_eval.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 8. 计算总训练步数
per_device_train_batch_size = 16  # 适当增大批次
num_train_epochs = 3  # 增加轮次
total_train_steps = (len(tokenized_train) // (per_device_train_batch_size * max(1, torch.cuda.device_count()))) * num_train_epochs
print(f"总训练步数: {total_train_steps}")



# 9. 修正 AdaLoRA 配置（基于实际模块名称）
ada_lora_config = AdaLoraConfig(
    task_type=TaskType.SEQ_CLS,
    init_r=8,
    target_r=4,
    beta1=0.85,
    beta2=0.85,
    tinit=max(20, total_train_steps // 10),
    tfinal=max(100, total_train_steps // 2),
    deltaT=max(10, total_train_steps // 50),
    lora_alpha=32,
    lora_dropout=0.1,
    # 基于实际模块名称的配置
    target_modules=[
        # 第0层
        "distilbert.transformer.layer.0.attention.q_lin",
        "distilbert.transformer.layer.0.attention.k_lin",
        "distilbert.transformer.layer.0.attention.v_lin",
        "distilbert.transformer.layer.0.attention.out_lin",
        "distilbert.transformer.layer.0.ffn.lin1",
        "distilbert.transformer.layer.0.ffn.lin2",
        # 第1层
        "distilbert.transformer.layer.1.attention.q_lin",
        "distilbert.transformer.layer.1.attention.k_lin",
        "distilbert.transformer.layer.1.attention.v_lin",
        "distilbert.transformer.layer.1.attention.out_lin",
        "distilbert.transformer.layer.1.ffn.lin1",
        "distilbert.transformer.layer.1.ffn.lin2",
        # 第2层
        "distilbert.transformer.layer.2.attention.q_lin",
        "distilbert.transformer.layer.2.attention.k_lin",
        "distilbert.transformer.layer.2.attention.v_lin",
        "distilbert.transformer.layer.2.attention.out_lin",
        "distilbert.transformer.layer.2.ffn.lin1",
        "distilbert.transformer.layer.2.ffn.lin2",
    ],
    total_step=total_train_steps
)

# 10. 创建 AdaLoRA 模型并验证
peft_model = get_peft_model(model, ada_lora_config)

print("可训练参数:")
peft_model.print_trainable_parameters()

print("\n模型中的 LoRA 适配器:")
for name, module in peft_model.named_modules():
    if "lora" in name.lower():
        print(name)



# 11. 数据收集器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 12. 评估函数（不变）
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

# 13. 训练参数（关键：降低学习率）
training_args = TrainingArguments(
    output_dir="./ada_lora_results_fixed",
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=32,
    num_train_epochs=num_train_epochs,
    learning_rate=2e-5,  # 预训练模型微调常用学习率（远小于之前的1e-3）
    warmup_ratio=0.1,
    weight_decay=0.01,  # 增加正则化
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
    fp16=True if device.type == "cuda" else False,
)

# 14. 训练
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("开始训练...")
trainer.train()

# 15. 最终评估
eval_results = trainer.evaluate()
print(f"最终评估准确率: {eval_results['eval_accuracy']:.4f}")  # 此时应在80%+

# 16. 保存适配器
peft_model.save_pretrained("./ada_lora_imdb_adapter_fixed")

# 17. 推理函数（不变）
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = peft_model(**inputs)
        logits = outputs.logits
    prob = torch.softmax(logits, dim=1)[0].cpu()
    return {"negative": prob[0].item(), "positive": prob[1].item()}

# 18. 测试预测（应正常区分正负）
print("\n测试结果：")
print("正面评论:", predict_sentiment("This movie was absolutely wonderful! The acting was great and the plot was engaging."))
print("负面评论:", predict_sentiment("Terrible experience, waste of time. The story was boring and the actors were bad."))

# 19. 奇异值统计（此时应输出结果）
def print_singular_values():
    print("\nAdaLoRA层奇异值统计:")
    has_sv = False
    for name, module in peft_model.named_modules():
        if hasattr(module, "sv") and module.sv is not None:
            has_sv = True
            sv = module.sv.detach().cpu().numpy()
            non_zero = np.sum(sv > 1e-6)
            print(f"{name}: 奇异值数量={len(sv)}, 非零={non_zero}, 最大={sv.max():.4f}")
    if not has_sv:
        print("仍未找到奇异值！请再次检查target_modules与模型结构是否匹配。")

print_singular_values()