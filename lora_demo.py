from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate
import numpy as np
# 添加国内镜像源
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ... existing code ...
# 1. 加载基础模型与分词器
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,  # 二分类（积极/消极）
    device_map="auto"  # 自动分配设备
)

# 2. 配置LoRA参数核心部分）
lora_config = LoraConfig(
    r=16,  # 低秩矩阵的秩，控制参数更新能力（通常8-32）
    lora_alpha=32,  # 缩放参数，α/r控制学习率缩放
    target_modules=[  # 指定需要注入LoRA的模块
        "query", "value"  # BERT注意力层的Q和V矩阵
    ],
    lora_dropout=0.05,  # Dropout概率
    bias="none",  # 不训练偏置参数
    task_type="SEQ_CLS",  # 任务类型
)

# 3. 应用LoRA适配器并冻结原模型参数
model = get_peft_model(model, lora_config)
# 打印参数训练状态
model.print_trainable_parameters()
# 输出示例: trainable params: 884,738 || all params: 109,514,850 || trainable%: 0.8078

# 4. 数据预处理
dataset = load_dataset("imdb")  # 加载IMDB影评数据集

def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

# 批量处理数据集
tokenized_dataset = dataset.map(preprocess_function, batched=True)
# 准备小样本子集用于演示（全量训练需移除select）
small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(100))

# 5. 定义评估指标
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 6. 训练配置
training_args = TrainingArguments(
    output_dir="./lora_bert_results",
    learning_rate=3e-4,  # LoRA通常使用比全量微调更高的学习率
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch",  # 已将evaluation_strategy改为eval_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 7. 初始化Trainer并开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# 8. 推理示例
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    return "positive" if predicted_class_id == 1 else "negative"

# 测试积极文本
print(predict_sentiment("This movie was amazing! The acting was superb and the plot was captivating."))
# 测试消极文本
print(predict_sentiment("Terrible film. The story was boring and the characters were uninteresting."))

# 9. 保存LoRA权重（仅保存适配器参数，约2-5MB）
model.save_pretrained("./lora_bert_sentiment")