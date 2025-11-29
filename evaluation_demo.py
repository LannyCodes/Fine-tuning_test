import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import jieba
import evaluate
import pandas as pd

# ==========================================
# 微调效果评估脚本
# ==========================================
# 该脚本演示了两种常用的微调效果评估方法：
# 1. 主观评估：Side-by-Side (SBS) 对比，人眼观察微调前后差异
# 2. 客观评估：使用 ROUGE 指标计算生成文本与标准答案的相似度
# ==========================================

MODEL_ID = "Qwen/Qwen2-7B-Instruct"
# 假设微调后的适配器保存在这里 (请根据实际情况修改)
ADAPTER_PATH = "/kaggle/working/qwen2_adalora_output" 

def load_models():
    """同时加载基座模型和微调后的模型"""
    print("正在加载基座模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # 加载基座模型 (为了演示，这里假设显存足够，或者分次加载)
    # 在实际显存受限环境下，建议先跑一次基座模型生成结果，保存后再跑微调模型
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # 加载微调后的模型 (Base + Adapter)
    # 注意：这里为了演示方便，直接在 base_model 上挂载 adapter
    # 实际对比时，你可能需要两个模型实例，或者加载/卸载 adapter
    print(f"正在加载 LoRA 适配器: {ADAPTER_PATH}...")
    try:
        finetuned_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    except Exception as e:
        print(f"警告: 无法加载适配器 ({e})，将仅使用基座模型演示。")
        finetuned_model = base_model

    return tokenizer, base_model, finetuned_model

def generate_response(model, tokenizer, instruction):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=128,
        do_sample=False  # 评估时通常关闭采样以保证确定性，或者开启采样测多样性
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def evaluate_subjective(tokenizer, base_model, finetuned_model):
    """
    方法一：Side-by-Side (SBS) 人工主观对比
    最直观的方法，直接看微调有没有学会特定的语气或知识
    """
    print("\n=== 1. Side-by-Side 主观评估 ===")
    test_cases = [
        "什么是 AdaLoRA？",  # 这是一个我们在微调数据里“注入”过知识的问题
        "解释量子纠缠。",    # 通用知识
        "你是谁？"          # 自我认知 (微调通常会修改这个)
    ]

    results = []
    for prompt in test_cases:
        print(f"\n[测试指令]: {prompt}")
        
        # 基座模型回答
        # (注: 如果是同一个模型实例，需要 disable adapter)
        with finetuned_model.disable_adapter():
            base_resp = generate_response(finetuned_model, tokenizer, prompt)
            
        # 微调模型回答
        ft_resp = generate_response(finetuned_model, tokenizer, prompt)
        
        print(f"--- 基座模型 ---\n{base_resp}")
        print(f"--- 微调模型 ---\n{ft_resp}")
        
        results.append({
            "Instruction": prompt,
            "Base Model": base_resp,
            "Fine-tuned": ft_resp
        })
    
    # 保存对比结果到 CSV 方便查看
    pd.DataFrame(results).to_csv("sbs_comparison.csv", index=False)
    print("\n对比结果已保存至 sbs_comparison.csv")

def evaluate_objective(tokenizer, model, references):
    """
    方法二：客观指标评估 (ROUGE-L)
    计算生成文本与参考答案的重合度
    注意：对于生成式任务，ROUGE 分数仅供参考，不代表绝对质量
    """
    print("\n=== 2. 客观指标评估 (ROUGE) ===")
    
    rouge = evaluate.load("rouge")
    
    predictions = []
    ground_truths = []
    
    print("正在生成回答并计算指标...")
    for item in references:
        prompt = item["instruction"]
        truth = item["output"]
        
        pred = generate_response(model, tokenizer, prompt)
        
        # 中文需要分词，否则 ROUGE 计算不准
        predictions.append(" ".join(jieba.cut(pred)))
        ground_truths.append(" ".join(jieba.cut(truth)))
    
    results = rouge.compute(predictions=predictions, references=ground_truths)
    print(f"ROUGE-1: {results['rouge1']:.4f} (词重合度)")
    print(f"ROUGE-L: {results['rougeL']:.4f} (最长公共子序列)")
    print("提示: ROUGE 分数越高，表示生成结果与标准答案的用词越接近。")

if __name__ == "__main__":
    # 1. 准备测试数据 (通常来自你的验证集)
    test_data = [
        {
            "instruction": "什么是 AdaLoRA？",
            "output": "AdaLoRA 是一种参数高效微调方法，它通过奇异值分解 (SVD) 动态分配不同层的参数预算，在训练过程中自适应地修剪不重要的秩。"
        },
        {
            "instruction": "Qwen2 7B 可以在 Kaggle 上跑吗？",
            "output": "可以，通过使用 4-bit 量化 (QLoRA) 和梯度累积，Qwen2 7B 可以在 Kaggle 的 T4/P100 16GB 显存上运行。"
        }
    ]

    # 2. 加载模型
    # 注意：如果在没有 GPU 的环境运行此脚本会非常慢或报错
    if torch.cuda.is_available():
        tokenizer, base_model, ft_model = load_models()
        
        # 3. 执行评估
        evaluate_subjective(tokenizer, base_model, ft_model)
        evaluate_objective(tokenizer, ft_model, test_data)
    else:
        print("未检测到 GPU，跳过实际加载模型。请在 GPU 环境下运行此脚本。")
