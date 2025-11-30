import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# 模型合并脚本 (LoRA Merge)
# ==========================================
# 功能：将微调后的 Adapter 权重合并回基座模型
# 输出：一个完整的、独立的模型文件夹，可直接用于 vLLM 等推理引擎
# ==========================================

# 1. 配置路径
MODEL_ID = "Qwen/Qwen2-7B-Instruct"
ADAPTER_PATH = "/kaggle/working/qwen2_spider_output"  # 你的微调输出路径
OUTPUT_PATH = "/kaggle/working/qwen2_spider_merged"   # 合并后的模型保存路径

def merge_model():
    print(f"正在加载基座模型: {MODEL_ID}")
    # 注意：合并时必须使用 float16 或 bfloat16，不能使用 4-bit 加载
    # 因为 merge_and_unload 不支持量化模型
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"显存不足警告: 合并模型需要加载完整 FP16 权重 (~15GB 显存)。\n错误信息: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print(f"正在加载 LoRA 适配器: {ADAPTER_PATH}")
    # 自动寻找最新的 checkpoint
    checkpoint_path = ADAPTER_PATH
    if os.path.exists(ADAPTER_PATH):
        subdirs = [d for d in os.listdir(ADAPTER_PATH) if d.startswith("checkpoint")]
        if subdirs:
            latest = sorted(subdirs, key=lambda x: int(x.split("-")[-1]))[-1]
            checkpoint_path = os.path.join(ADAPTER_PATH, latest)
            print(f"使用 Checkpoint: {checkpoint_path}")

    model = PeftModel.from_pretrained(base_model, checkpoint_path)

    print("开始合并权重 (Merge and Unload)...")
    # 核心步骤：将 LoRA 权重加到基座模型权重中
    merged_model = model.merge_and_unload()

    print(f"正在保存合并后的模型至: {OUTPUT_PATH} ...")
    merged_model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_PATH)
    
    print("=== 合并完成 ===")
    print(f"你可以直接使用 {OUTPUT_PATH} 进行 vLLM 部署")

if __name__ == "__main__":
    # 检查显存是否足够 (合并需要加载完整模型)
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"检测到 GPU 显存: {gpu_mem:.2f} GB")
        if gpu_mem < 16:
            print("警告: 显存可能不足以执行 FP16 合并 (推荐 >16GB)。尝试在 CPU 上运行或使用高显存环境。")
    
    merge_model()
