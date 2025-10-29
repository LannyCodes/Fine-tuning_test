# AdaLoRA参数调优指南和自动调优脚本

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
import itertools
import json
from datetime import datetime
import os

# 1. 理论指导原则
class AdaLoRAParameterGuide:
    """AdaLoRA参数选择指导"""
    
    @staticmethod
    def get_recommended_params(model_size, task_complexity, dataset_size):
        """
        根据模型大小、任务复杂度、数据集大小推荐参数
        
        Args:
            model_size: "small" (<100M), "medium" (100M-1B), "large" (>1B)
            task_complexity: "simple" (分类), "medium" (NER), "complex" (生成)
            dataset_size: "small" (<10K), "medium" (10K-100K), "large" (>100K)
        """
        
        # 基础参数映射表
        base_params = {
            "small": {"init_r": 8, "target_r": 4, "alpha": 16},
            "medium": {"init_r": 16, "target_r": 8, "alpha": 32}, 
            "large": {"init_r": 32, "target_r": 16, "alpha": 64}
        }
        
        # 任务复杂度调整
        complexity_multiplier = {
            "simple": 0.8,
            "medium": 1.0,
            "complex": 1.5
        }
        
        # 数据集大小调整
        data_multiplier = {
            "small": 0.7,
            "medium": 1.0,
            "large": 1.2
        }
        
        base = base_params[model_size]
        task_mult = complexity_multiplier[task_complexity]
        data_mult = data_multiplier[dataset_size]
        
        # 计算推荐参数
        recommended = {
            "init_r": max(4, int(base["init_r"] * task_mult * data_mult)),
            "target_r": max(2, int(base["target_r"] * task_mult * data_mult)),
            "lora_alpha": max(8, int(base["alpha"] * task_mult))
        }
        
        # 确保 target_r <= init_r
        recommended["target_r"] = min(recommended["target_r"], recommended["init_r"] // 2)
        
        return recommended

# 2. 自动参数搜索网格
def create_param_grid():
    """创建参数搜索网格"""
    
    param_grid = {
        # 秩参数 - 核心参数
        "init_r": [4, 8, 16, 32],
        "target_r": [2, 4, 8, 16],  # 会根据init_r动态调整
        
        # 学习相关参数
        "lora_alpha": [8, 16, 32, 64],
        "lora_dropout": [0.05, 0.1, 0.15],
        
        # AdaLoRA特有参数
        "beta1": [0.8, 0.85, 0.9],
        "beta2": [0.8, 0.85, 0.9],
        
        # 时间调度参数（相对于总步数的比例）
        "tinit_ratio": [0.1, 0.15, 0.2],    # 热身阶段比例
        "tfinal_ratio": [0.6, 0.7, 0.8],    # 调整阶段结束比例
        "deltaT_ratio": [0.02, 0.05, 0.1]   # 更新频率比例
    }
    
    return param_grid

# 3. 智能参数组合生成器
def generate_valid_combinations(param_grid, max_combinations=20):
    """生成有效的参数组合，避免无效配置"""
    
    combinations = []
    
    # 基础组合
    base_combinations = [
        # 保守配置 - 适用于小数据集
        {"init_r": 8, "target_r": 4, "lora_alpha": 16, "lora_dropout": 0.1},
        # 标准配置 - 适用于中等数据集  
        {"init_r": 16, "target_r": 8, "lora_alpha": 32, "lora_dropout": 0.05},
        # 激进配置 - 适用于大数据集
        {"init_r": 32, "target_r": 16, "lora_alpha": 64, "lora_dropout": 0.05},
    ]
    
    for base in base_combinations:
        for beta1 in param_grid["beta1"]:
            for beta2 in param_grid["beta2"]:
                for tinit_ratio in param_grid["tinit_ratio"]:
                    for tfinal_ratio in param_grid["tfinal_ratio"]:
                        if tfinal_ratio > tinit_ratio:  # 确保时间顺序正确
                            for deltaT_ratio in param_grid["deltaT_ratio"]:
                                combination = base.copy()
                                combination.update({
                                    "beta1": beta1,
                                    "beta2": beta2, 
                                    "tinit_ratio": tinit_ratio,
                                    "tfinal_ratio": tfinal_ratio,
                                    "deltaT_ratio": deltaT_ratio
                                })
                                combinations.append(combination)
                                
                                if len(combinations) >= max_combinations:
                                    return combinations
    
    return combinations

# 4. 参数验证函数
def validate_params(params, total_steps):
    """验证参数配置的合理性"""
    
    errors = []
    
    # 秩参数检查
    if params["target_r"] >= params["init_r"]:
        errors.append("target_r应该小于init_r")
    
    if params["target_r"] < 2:
        errors.append("target_r不应小于2")
        
    if params["init_r"] > 64:
        errors.append("init_r过大，建议不超过64")
    
    # Alpha参数检查
    if params["lora_alpha"] < params["init_r"]:
        errors.append("lora_alpha通常应大于等于init_r")
    
    # Beta参数检查
    if params["beta1"] >= 1.0 or params["beta2"] >= 1.0:
        errors.append("beta1和beta2应小于1.0")
    
    # 时间参数检查
    tinit = int(params["tinit_ratio"] * total_steps)
    tfinal = int(params["tfinal_ratio"] * total_steps)
    deltaT = max(1, int(params["deltaT_ratio"] * total_steps))
    
    if tinit >= tfinal:
        errors.append("tinit应小于tfinal")
        
    if tfinal >= total_steps:
        errors.append("tfinal应小于总训练步数")
        
    if deltaT > (tfinal - tinit) // 2:
        errors.append("deltaT相对于调整窗口过大")
    
    return errors, {"tinit": tinit, "tfinal": tfinal, "deltaT": deltaT}

# 5. 单次训练评估函数
def evaluate_single_config(params, train_dataset, eval_dataset, tokenizer, total_steps):
    """评估单个参数配置"""
    
    print(f"\n测试配置: init_r={params['init_r']}, target_r={params['target_r']}, alpha={params['lora_alpha']}")
    
    try:
        # 验证参数
        errors, time_params = validate_params(params, total_steps)
        if errors:
            print(f"参数验证失败: {errors}")
            return None
        
        # 创建模型
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=2
        )
        
        # 配置AdaLoRA
        ada_config = AdaLoraConfig(
            task_type=TaskType.SEQ_CLS,
            init_r=params["init_r"],
            target_r=params["target_r"],
            beta1=params["beta1"],
            beta2=params["beta2"],
            tinit=time_params["tinit"],
            tfinal=time_params["tfinal"],
            deltaT=time_params["deltaT"],
            lora_alpha=params["lora_alpha"],
            lora_dropout=params["lora_dropout"],
            target_modules=["q_lin", "v_lin"],
            total_step=total_steps
        )
        
        model = get_peft_model(model, ada_config)
        
        # 训练配置（快速评估）
        training_args = TrainingArguments(
            output_dir=f"./temp_results_{datetime.now().strftime('%H%M%S')}",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            num_train_epochs=1,  # 只训练1轮进行快速评估
            learning_rate=2e-5,
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="no",  # 不保存模型
            report_to="none",
        )
        
        # 训练器
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        )
        
        # 训练
        trainer.train()
        
        # 评估
        eval_results = trainer.evaluate()
        
        # 计算可训练参数数量
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        result = {
            "params": params,
            "accuracy": eval_results.get("eval_loss", float('inf')),  # 使用loss作为主要指标
            "trainable_params": trainable_params,
            "efficiency": trainable_params / total_params,
            "time_params": time_params
        }
        
        print(f"结果: loss={result['accuracy']:.4f}, 可训练参数={trainable_params:,}")
        
        # 清理临时文件
        import shutil
        if os.path.exists(training_args.output_dir):
            shutil.rmtree(training_args.output_dir)
            
        return result
        
    except Exception as e:
        print(f"配置评估失败: {e}")
        return None

# 6. 自动调优主函数
def auto_tune_adalora(max_trials=10):
    """自动调优AdaLoRA参数"""
    
    print("=== AdaLoRA参数自动调优 ===")
    
    # 准备数据
    print("准备数据集...")
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # 小样本用于快速调优
    train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
    eval_dataset = dataset["test"].shuffle(seed=42).select(range(200))
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    train_dataset = train_dataset.rename_column("label", "labels").remove_columns(["text"])
    eval_dataset = eval_dataset.rename_column("label", "labels").remove_columns(["text"])
    
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")
    
    # 计算总步数
    total_steps = len(train_dataset) // 16  # batch_size=16
    
    # 生成参数组合
    param_grid = create_param_grid()
    combinations = generate_valid_combinations(param_grid, max_trials)
    
    print(f"将测试 {len(combinations)} 个参数组合")
    
    # 评估所有组合
    results = []
    for i, params in enumerate(combinations):
        print(f"\n进度: {i+1}/{len(combinations)}")
        result = evaluate_single_config(params, train_dataset, eval_dataset, tokenizer, total_steps)
        if result:
            results.append(result)
    
    # 排序结果（按loss升序）
    results.sort(key=lambda x: x["accuracy"])
    
    # 输出最佳配置
    print("\n=== 调优结果 ===")
    print("前5个最佳配置:")
    
    for i, result in enumerate(results[:5]):
        params = result["params"]
        print(f"\n第{i+1}名:")
        print(f"  init_r: {params['init_r']}, target_r: {params['target_r']}")
        print(f"  lora_alpha: {params['lora_alpha']}, dropout: {params['lora_dropout']}")
        print(f"  beta1: {params['beta1']}, beta2: {params['beta2']}")
        print(f"  loss: {result['accuracy']:.4f}")
        print(f"  可训练参数: {result['trainable_params']:,}")
        print(f"  参数效率: {result['efficiency']*100:.2f}%")
    
    # 保存结果
    with open("adalora_tuning_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n完整结果已保存到: adalora_tuning_results.json")
    
    return results[0] if results else None

# 7. 参数分析和可视化
def analyze_param_sensitivity():
    """分析参数敏感性"""
    
    print("\n=== AdaLoRA参数敏感性分析 ===")
    
    sensitivity_analysis = {
        "init_r": {
            "description": "初始秩的影响",
            "recommendations": {
                "小模型(<100M)": "4-8",
                "中等模型(100M-1B)": "8-16", 
                "大模型(>1B)": "16-32"
            },
            "trade_offs": "更高的init_r提供更强的表达能力，但增加计算开销"
        },
        
        "target_r": {
            "description": "目标秩的影响",
            "recommendations": {
                "简单任务": "init_r的1/4到1/2",
                "复杂任务": "init_r的1/2到2/3"
            },
            "trade_offs": "target_r太小可能欠拟合，太大则失去压缩优势"
        },
        
        "lora_alpha": {
            "description": "缩放因子的影响",
            "recommendations": {
                "一般设置": "init_r的1-4倍",
                "稳定训练": "2*init_r"
            },
            "trade_offs": "控制LoRA贡献的幅度，影响训练稳定性"
        },
        
        "beta1/beta2": {
            "description": "重要性评分的动量参数",
            "recommendations": {
                "标准设置": "0.85",
                "快速适应": "0.8",
                "稳定适应": "0.9"
            },
            "trade_offs": "影响秩调整的响应速度"
        }
    }
    
    for param, info in sensitivity_analysis.items():
        print(f"\n{param.upper()}:")
        print(f"  说明: {info['description']}")
        print(f"  权衡: {info['trade_offs']}")
        print("  推荐设置:")
        for scenario, value in info['recommendations'].items():
            print(f"    {scenario}: {value}")

if __name__ == "__main__":
    # 1. 显示理论指导
    print("=== AdaLoRA参数理论指导 ===")
    guide = AdaLoRAParameterGuide()
    
    # 示例推荐
    examples = [
        ("small", "simple", "small"),    # 小模型，简单任务，小数据
        ("medium", "medium", "medium"),  # 中等模型，中等任务，中等数据
        ("large", "complex", "large")    # 大模型，复杂任务，大数据
    ]
    
    for model_size, task_complexity, dataset_size in examples:
        params = guide.get_recommended_params(model_size, task_complexity, dataset_size)
        print(f"\n{model_size}模型 + {task_complexity}任务 + {dataset_size}数据:")
        print(f"  推荐参数: {params}")
    
    # 2. 参数敏感性分析
    analyze_param_sensitivity()
    
    # 3. 询问是否运行自动调优
    response = input("\n是否运行自动参数调优? (y/n): ")
    if response.lower() == 'y':
        best_config = auto_tune_adalora(max_trials=15)
        if best_config:
            print(f"\n最佳配置已找到: {best_config['params']}")
    
    print("\n参数调优指南运行完成！")