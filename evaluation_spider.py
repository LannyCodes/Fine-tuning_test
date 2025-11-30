import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==========================================
# Spider NL2SQL 评估脚本
# ==========================================
# 该脚本用于评估微调后的模型在 Spider 开发集 (dev.json) 上的表现。
# 评估指标：String-Match Accuracy (生成的 SQL 与 标准答案 SQL 的字符串匹配度)
# ==========================================

MODEL_ID = "Qwen/Qwen2-7B-Instruct"
# 微调后的适配器路径
ADAPTER_PATH = "/kaggle/working/qwen2_spider_output" 
# Spider 数据集路径
DATA_DIR = "/kaggle/input/spider-lora/data"

def load_spider_tables(tables_path):
    """
    读取 tables.json 并构建 db_id -> schema_text 的映射
    (逻辑与训练脚本保持完全一致)
    """
    print(f"正在加载 Schema 信息: {tables_path}")
    if not os.path.exists(tables_path):
        print(f"文件不存在: {tables_path}")
        return {}

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

def load_spider_dev_data(data_path, db_schema_map):
    """加载测试数据 (dev.json)"""
    print(f"正在加载测试数据: {data_path}")
    if not os.path.exists(data_path):
        print(f"文件不存在: {data_path}")
        return []
        
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    test_data = []
    for item in raw_data:
        db_id = item['db_id']
        question = item['question']
        query = item['query']
        schema_context = db_schema_map.get(db_id, "")
        
        # 构造与训练时一致的 Prompt
        instruction = (
            f"Human: 你是一个专业的数据库工程师。请根据以下的数据库 Schema，编写对应的 SQL 语句来回答用户的问题。\n\n"
            f"[Schema 信息]\n{schema_context}\n\n"
            f"用户问题: {question}\n"
            f"Assistant: "
        )
        
        test_data.append({
            "instruction": instruction,
            "output": query,
            "question_raw": question,
            "db_id": db_id
        })
    return test_data

def load_models():
    """加载模型 (4-bit 量化)"""
    print("正在加载基座模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"正在加载 LoRA 适配器: {ADAPTER_PATH}...")
    try:
        # 检查是否存在 Checkpoint
        checkpoint_path = None
        if os.path.exists(ADAPTER_PATH):
            subdirs = [d for d in os.listdir(ADAPTER_PATH) if d.startswith("checkpoint")]
            if subdirs:
                # 找到最新的 checkpoint
                latest_checkpoint = sorted(subdirs, key=lambda x: int(x.split("-")[-1]))[-1]
                checkpoint_path = os.path.join(ADAPTER_PATH, latest_checkpoint)
                print(f"发现 Checkpoint: {checkpoint_path}")
        
        if checkpoint_path:
             finetuned_model = PeftModel.from_pretrained(base_model, checkpoint_path)
        else:
             finetuned_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
             
    except Exception as e:
        print(f"警告: 加载适配器失败 ({e})，将使用基座模型进行测试。")
        finetuned_model = base_model

    return tokenizer, finetuned_model

def generate_response(model, tokenizer, instruction):
    """生成回答"""
    model_inputs = tokenizer([instruction], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=128, # SQL 通常不会特别长
        do_sample=False,    # 确定性生成
        pad_token_id=tokenizer.pad_token_id
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def evaluate_spider(tokenizer, model, test_data, limit=None):
    """执行批量评估"""
    print(f"\n=== 开始 Spider 评估 (共 {len(test_data)} 条) ===")
    
    if limit:
        test_data = test_data[:limit]
        print(f"仅测试前 {limit} 条数据")
    
    results = []
    correct_count = 0
    
    try:
        from tqdm import tqdm
        iterator = tqdm(test_data, desc="Generating SQL")
    except ImportError:
        iterator = test_data
    
    for i, item in enumerate(iterator):
        prompt = item["instruction"]
        truth = item["output"]
        
        pred_sql = generate_response(model, tokenizer, prompt)
        pred_sql = pred_sql.strip()
        
        # 简单的准确率计算 (Exact Match - String Level)
        # 忽略大小写和末尾分号差异
        clean_pred = pred_sql.lower().strip().rstrip(";")
        clean_truth = truth.lower().strip().rstrip(";")
        
        is_match = (clean_pred == clean_truth)
        if is_match:
            correct_count += 1
        
        results.append({
            "db_id": item['db_id'],
            "question": item['question_raw'],
            "ground_truth": truth,
            "prediction": pred_sql,
            "is_match": is_match
        })
        
        if not hasattr(iterator, "desc") and (i + 1) % 10 == 0:
            print(f"已处理: {i + 1}/{len(test_data)}")
    
    accuracy = correct_count / len(test_data) if len(test_data) > 0 else 0
    print(f"\nString-Match Accuracy: {accuracy:.4f} ({correct_count}/{len(test_data)})")
    
    output_csv = "spider_evaluation_results.csv"
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"详细结果已保存至: {output_csv}")

if __name__ == "__main__":
    # 检查 GPU
    if not torch.cuda.is_available():
        print("警告: 未检测到 GPU，推理将非常慢或无法运行。")
    
    # 1. 准备数据
    tables_path = os.path.join(DATA_DIR, "tables.json")
    dev_path = os.path.join(DATA_DIR, "dev.json")
    
    # 本地调试用的 Mock 数据 (如果找不到真实数据)
    if not os.path.exists(tables_path):
        print("未找到 Spider 数据集，使用 Mock 数据进行测试演示...")
        # 创建一些假的 Schema 和数据用于演示代码逻辑是否跑通
        mock_db_map = {
            "mock_db": "Table: users\nColumns: id, name, age\n\nTable: orders\nColumns: id, user_id, amount"
        }
        test_data = [
            {
                "instruction": f"Human: 你是一个专业的数据库工程师。请根据以下的数据库 Schema，编写对应的 SQL 语句来回答用户的问题。\n\n[Schema 信息]\n{mock_db_map['mock_db']}\n\n用户问题: 查询所有用户的名字\nAssistant: ",
                "output": "SELECT name FROM users",
                "question_raw": "查询所有用户的名字",
                "db_id": "mock_db"
            }
        ]
    else:
        db_map = load_spider_tables(tables_path)
        test_data = load_spider_dev_data(dev_path, db_map)
    
    # 2. 加载模型
    tokenizer, model = load_models()
    
    # 3. 评估 (可以设置 limit=50 快速验证)
    # 全量评估可能需要较长时间
    evaluate_spider(tokenizer, model, test_data, limit=20) 
