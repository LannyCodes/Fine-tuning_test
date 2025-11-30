# ==========================================
# Kaggle Qwen2-7B AdaLoRA Spider NL2SQL 微调脚本
# ==========================================

import os
# 移除强制单卡限制，允许使用多卡
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import torch
from contextlib import contextmanager
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq, # 使用 Seq2Seq Collator 更适合生成任务
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, # 切换为标准 LoRA
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)

# 强制单卡
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" (已移至顶部)

# --- 配置部分 ---
MODEL_ID = "Qwen/Qwen2-7B-Instruct"
OUTPUT_DIR = "/kaggle/working/qwen2_spider_output"
DATA_DIR = "/kaggle/input/spider-lora/data/spider"  # Spider 数据集路径

# --- Spider 数据处理工具函数 ---

def load_spider_tables(tables_path):
    """
    读取 tables.json 并构建 db_id -> schema_text 的映射
    """
    print(f"正在加载 Schema 信息: {tables_path}")
    with open(tables_path, 'r', encoding='utf-8') as f:
        tables_data = json.load(f)
        
    db_schema_map = {}
    
    for db in tables_data:
        db_id = db['db_id']
        table_names = db['table_names_original']
        column_names = db['column_names_original'] # [table_idx, col_name]
        
        # 整理每个表的列
        schema_dict = {t_name: [] for t_name in table_names}
        
        for table_idx, col_name in column_names:
            if table_idx == -1: # 通用列如 *
                continue
            if table_idx < len(table_names):
                table_name = table_names[table_idx]
                schema_dict[table_name].append(col_name)
                
        # 格式化为文本
        schema_text = ""
        for t_name, cols in schema_dict.items():
            schema_text += f"Table: {t_name}\nColumns: {', '.join(cols)}\n\n"
            
        db_schema_map[db_id] = schema_text.strip()
        
    return db_schema_map

def load_spider_data(data_path, db_schema_map):
    """
    加载训练数据 (train_spider.json) 并结合 Schema
    """
    print(f"正在加载训练数据: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    processed_data = []
    for item in raw_data:
        db_id = item['db_id']
        question = item['question']
        query = item['query']
        
        # 获取对应的 schema
        schema_context = db_schema_map.get(db_id, "")
        
        processed_data.append({
            "instruction": question,
            "output": query,
            "schema": schema_context
        })
        
    return processed_data

@contextmanager
def torch_main_process_first(local_rank: int, desc: str = "process"):
    """
    Context manager that ensures the main process runs the block first, while other processes wait.
    Copied from transformers.trainer_pt_utils
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()

# --- 主训练函数 ---

def train_spider():
    print(f"正在检查环境...")
    if not torch.cuda.is_available():
        print("警告: 未检测到 GPU，代码可能无法运行。")
    
    # 1. 准备数据
    tables_path = os.path.join(DATA_DIR, "tables.json")
    train_path = os.path.join(DATA_DIR, "train_spider.json")
    
    # 如果找不到文件，尝试使用 dev.json 或者报错 (为了兼容性)
    if not os.path.exists(train_path):
        print(f"未找到 {train_path}，尝试查找目录...")
        # 简单的容错逻辑，列出目录下文件帮助调试
        if os.path.exists(DATA_DIR):
            print(f"目录内容: {os.listdir(DATA_DIR)}")
        else:
            print(f"目录不存在: {DATA_DIR}")
        return

    # 加载并处理数据
    db_map = load_spider_tables(tables_path)
    train_data_list = load_spider_data(train_path, db_map)
    
    # 仅使用前 500 条数据
    # if len(train_data_list) > 500:
    #     train_data_list = train_data_list[:500]
    #     print(f"为了加速训练，已截取前 500 条数据 (原数据量: {len(load_spider_data(train_path, db_map))})")
    
    # 使用全部数据
    print(f"使用全部训练数据: {len(train_data_list)} 条")
    
    # 转为 HuggingFace Dataset
    full_dataset = Dataset.from_list(train_data_list)
    print(f"总数据量: {len(full_dataset)}")
    
    # 划分验证集 (取 5% 做验证，Spider 数据集不大，尽量多留给训练)
    split_dataset = full_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # 2. 模型与 Tokenizer
    compute_dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # 确保 padding 在右侧 (生成任务要求)

    # DDP 环境下 device_map 配置逻辑
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    
    # 预先定义 gradient_accumulation_steps 初始值
    gradient_accumulation_steps = 8
    
    if ddp:
        # 必须先初始化进程组，否则后面的 barrier() 会报错
        if not torch.distributed.is_initialized():
             torch.distributed.init_process_group(backend="nccl")
        
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        print(f"检测到 DDP 环境 (World Size: {world_size})，已调整 device_map 和梯度累积步数")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 3. 数据预处理 (Tokenization)
    def process_func(example):
        # 构建 Prompt: 包含 Schema 信息
        prompt = (
            f"Human: 你是一个专业的数据库工程师。请根据以下的数据库 Schema，编写对应的 SQL 语句来回答用户的问题。\n\n"
            f"[Schema 信息]\n{example['schema']}\n\n"
            f"用户问题: {example['instruction']}\n"
            f"Assistant: "
        )
        
        # Qwen2 对话模板格式化 (可选，或者直接拼接)
        # 这里直接拼接简单明了
        
        inputs = tokenizer(
            prompt + example['output'] + tokenizer.eos_token, 
            truncation=True, 
            max_length=1024, # 增加长度以容纳 Schema
            padding="max_length"
        )
        
        inputs["labels"] = inputs["input_ids"].copy()
        
        # Mask 掉 Prompt 部分的 Loss
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids)
        if prompt_len < 1024:
            inputs["labels"][:prompt_len] = [-100] * prompt_len
            
        return inputs

    print("正在对数据进行 Tokenization (这可能需要几分钟)...")
    # 在 DDP 模式下，只在主进程进行 map 操作，并使用 load_from_cache_file=True
    with torch_main_process_first(local_rank=int(os.environ.get("LOCAL_RANK", -1)), desc="dataset map"):
        train_dataset = train_dataset.map(process_func, remove_columns=train_dataset.column_names)
        eval_dataset = eval_dataset.map(process_func, remove_columns=eval_dataset.column_names)

    # 4. LoRA 配置 (替换 AdaLoRA)
    # 计算步数
    # DDP 模式下，per_device_batch_size 决定了每张卡的显存占用
    # T4 (16GB) 在 4-bit QLoRA 下，可以尝试 2 或 4
    per_device_batch_size = 2 
    # gradient_accumulation_steps 在上面已经定义并根据 DDP 调整过了
    
    num_epochs = 2 # Spider 数据集较复杂，2-3 个 epoch
    
    num_update_steps_per_epoch = len(train_dataset) // (per_device_batch_size * gradient_accumulation_steps * world_size)
    total_train_steps = num_update_steps_per_epoch * num_epochs
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,                # 标准 LoRA 秩，Spider 任务建议设大一点
        lora_alpha=128,      # 通常是 r 的 2 倍
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. 训练参数
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=20,
        num_train_epochs=num_epochs,
        learning_rate=2e-4, # LoRA 常用 2e-4
        fp16=True,
        optim="paged_adamw_8bit",
        eval_strategy="steps",
        eval_steps=500, # 减少评估频率，避免频繁打断训练
        save_strategy="steps",
        save_steps=500, # 减少保存频率，节省磁盘空间
        load_best_model_at_end=True, # DDP 下这可能会比较慢，但为了效果保留
        metric_for_best_model="eval_loss",
        report_to="none",
        ddp_find_unused_parameters=False # DDP 关键参数
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    print("开始训练...")
    # 自动断点恢复逻辑
    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        from transformers.trainer_utils import get_last_checkpoint
        last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
        
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    print(f"保存模型到 {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("训练完成！")

if __name__ == "__main__":
    train_spider()
