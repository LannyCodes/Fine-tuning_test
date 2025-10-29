# 快速参数搜索脚本
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import AdaLoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import numpy as np

def quick_param_search():
    """快速参数搜索，找到大致范围"""
    
    # 候选参数组合
    param_candidates = [
        # (init_r, target_r, lora_alpha)
        (4, 2, 8),      # 最小配置
        (8, 4, 16),     # 小配置
        (16, 8, 32),    # 标准配置  
        (32, 16, 64),   # 大配置
        (16, 4, 32),    # 高压缩比
        (24, 12, 48),   # 中等配置
    ]
    
    print("=== 快速参数搜索 ===")
    
    # 准备小样本数据（500样本，快速验证）
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    small_train = dataset["train"].shuffle(seed=42).select(range(500))
    small_eval = dataset["test"].shuffle(seed=42).select(range(100))
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")
    
    train_dataset = small_train.map(tokenize_function, batched=True)
    eval_dataset = small_eval.map(tokenize_function, batched=True)
    
    train_dataset = train_dataset.rename_column("label", "labels").remove_columns(["text"])
    eval_dataset = eval_dataset.rename_column("label", "labels").remove_columns(["text"])
    
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")
    
    results = []
    
    for i, (init_r, target_r, lora_alpha) in enumerate(param_candidates):
        print(f"\n测试组合 {i+1}/{len(param_candidates)}: init_r={init_r}, target_r={target_r}, alpha={lora_alpha}")
        
        try:
            # 创建模型
            model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
            
            # AdaLoRA配置
            config = AdaLoraConfig(
                task_type=TaskType.SEQ_CLS,
                init_r=init_r,
                target_r=target_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.1,
                target_modules=["q_lin", "v_lin"],
                beta1=0.85,
                beta2=0.85,
                tinit=10,
                tfinal=30,
                deltaT=5,
                total_step=50  # 短训练
            )
            
            model = get_peft_model(model, config)
            
            # 快速训练（只训练几步）
            training_args = TrainingArguments(
                output_dir=f"./temp_{init_r}_{target_r}",
                per_device_train_batch_size=16,
                num_train_epochs=1,
                max_steps=50,  # 只训练50步
                learning_rate=2e-5,
                logging_steps=25,
                eval_strategy="steps",
                eval_steps=25,
                save_strategy="no",
                report_to="none",
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )
            
            # 训练和评估
            trainer.train()
            eval_result = trainer.evaluate()
            
            # 统计参数
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            result = {
                "init_r": init_r,
                "target_r": target_r, 
                "lora_alpha": lora_alpha,
                "eval_loss": eval_result["eval_loss"],
                "trainable_params": trainable_params,
                "compression_ratio": target_r / init_r
            }
            
            results.append(result)
            print(f"  结果: loss={result['eval_loss']:.4f}, 参数={trainable_params:,}")
            
        except Exception as e:
            print(f"  失败: {e}")
    
    # 排序并显示结果
    results.sort(key=lambda x: x["eval_loss"])
    
    print("\n=== 快速搜索结果 (按loss排序) ===")
    for i, result in enumerate(results[:3]):
        print(f"第{i+1}名: init_r={result['init_r']}, target_r={result['target_r']}, "
              f"alpha={result['lora_alpha']}, loss={result['eval_loss']:.4f}")
    
    return results[0] if results else None

if __name__ == "__main__":
    best_config = quick_param_search()
    if best_config:
        print(f"\n推荐配置: {best_config}")