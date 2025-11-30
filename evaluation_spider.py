import os
import json
import torch
import sqlite3
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==========================================
# Spider NL2SQL 评估脚本
# ==========================================
# 该脚本用于评估微调后的模型在 Spider 开发集 (dev.json) 上的表现。
# 评估指标：Execution Accuracy (在 SQLite 数据库中执行并对比结果)
# ==========================================

MODEL_ID = "Qwen/Qwen2-7B-Instruct"
# 微调后的适配器路径
ADAPTER_PATH = "/kaggle/working/qwen2_spider_output" 
# Spider 数据集路径 (需要包含 database 文件夹)
DATA_DIR = "/kaggle/input/spider-lora/data/spider"
DB_DIR = os.path.join(DATA_DIR, "database") # 数据库文件所在目录

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

def load_spider_dev_data(data_path, db_schema_map):
    """
    加载测试数据 (dev.json) 并结合 Schema
    """
    print(f"正在加载测试数据: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    processed_data = []
    for item in raw_data:
        db_id = item['db_id']
        question = item['question']
        query = item['query'] # Ground Truth SQL
        
        # 获取对应的 schema
        schema_context = db_schema_map.get(db_id, "")
        
        # 构建与训练时一致的 Prompt
        instruction = (
            f"Human: 你是一个专业的数据库工程师。请根据以下的数据库 Schema，编写对应的 SQL 语句来回答用户的问题。\n\n"
            f"[Schema 信息]\n{schema_context}\n\n"
            f"用户问题: {question}\n"
            f"Assistant: "
        )
        
        processed_data.append({
            "instruction": instruction,
            "output": query,
            "question_raw": question,
            "db_id": db_id
        })
        
    return processed_data

def load_models():
    """加载基座模型和 LoRA 适配器"""
    print(f"正在加载基座模型: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    if os.path.exists(ADAPTER_PATH):
        print(f"正在加载适配器: {ADAPTER_PATH}")
        # 自动查找最新的 checkpoint
        checkpoint_path = ADAPTER_PATH
        subdirs = [d for d in os.listdir(ADAPTER_PATH) if d.startswith("checkpoint")]
        if subdirs:
            latest = sorted(subdirs, key=lambda x: int(x.split("-")[-1]))[-1]
            checkpoint_path = os.path.join(ADAPTER_PATH, latest)
            print(f"使用 Checkpoint: {checkpoint_path}")
            
        model = PeftModel.from_pretrained(model, checkpoint_path)
    else:
        print("警告: 未找到适配器路径，将使用原始基座模型进行推理！")
        
    model.eval()
    return tokenizer, model

def generate_response(model, tokenizer, prompt):
    """生成 SQL 回答"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128, # SQL 通常不会特别长
            do_sample=False,    # 确定性生成，适合代码生成
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response

def execute_sql(db_path, sql):
    """在 SQLite 中执行 SQL 并返回结果"""
    if not os.path.exists(db_path):
        return "Error: DB not found"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.close()
        # 将结果转为集合以便对比（忽略顺序）
        return set(result)
    except Exception as e:
        return f"Error: {str(e)}"

def evaluate_spider(tokenizer, model, test_data, limit=None):
    """执行批量评估 (Execution Accuracy)"""
    print(f"\n=== 开始 Spider 评估 (共 {len(test_data)} 条) ===")
    
    if limit:
        test_data = test_data[:limit]
        print(f"仅测试前 {limit} 条数据")
    
    results = []
    exec_correct_count = 0
    str_correct_count = 0
    valid_sql_count = 0
    
    try:
        from tqdm import tqdm
        iterator = tqdm(test_data, desc="Generating SQL")
    except ImportError:
        iterator = test_data
    
    for i, item in enumerate(iterator):
        prompt = item["instruction"]
        truth_sql = item["output"]
        db_id = item["db_id"]
        
        # 1. 生成 SQL
        pred_sql = generate_response(model, tokenizer, prompt)
        pred_sql = pred_sql.strip()
        
        # 2. 字符串匹配 (String Match)
        clean_pred = pred_sql.lower().strip().rstrip(";")
        clean_truth = truth_sql.lower().strip().rstrip(";")
        is_str_match = (clean_pred == clean_truth)
        if is_str_match:
            str_correct_count += 1
            
        # 3. 执行匹配 (Execution Match)
        db_path = os.path.join(DB_DIR, db_id, f"{db_id}.sqlite")
        
        truth_result = execute_sql(db_path, truth_sql)
        pred_result = execute_sql(db_path, pred_sql)
        
        # 统计语法合法的 SQL (Valid SQL)
        # 只要不报错（返回的不是 Error 字符串），就认为语法合法（即便结果不对）
        is_valid_sql = not isinstance(pred_result, str)
        if is_valid_sql:
            valid_sql_count += 1

        # 判断执行结果是否一致
        # 注意：如果两个都报错且报错信息一样，不算正确；只有成功执行且结果一致才算
        is_exec_match = False
        if not isinstance(truth_result, str) and not isinstance(pred_result, str):
            if truth_result == pred_result:
                is_exec_match = True
                exec_correct_count += 1
        
        results.append({
            "db_id": db_id,
            "question": item['question_raw'],
            "ground_truth": truth_sql,
            "prediction": pred_sql,
            "is_str_match": is_str_match,
            "is_exec_match": is_exec_match,
            "is_valid_sql": is_valid_sql,
            "exec_error": pred_result if isinstance(pred_result, str) else ""
        })
        
        # 打印前几个错误样例，方便调试
        if not is_valid_sql and i < 5:
            print(f"\n[Error Sample {i}]")
            print(f"Prompt: {prompt[-200:]}...") # 只打印最后 200 字符
            print(f"Prediction: {pred_sql}")
            print(f"Error: {pred_result}")
            print("-" * 30)
        
        if not hasattr(iterator, "desc") and (i + 1) % 10 == 0:
            print(f"已处理: {i + 1}/{len(test_data)}")
    
    total = len(test_data)
    exec_acc = exec_correct_count / total if total > 0 else 0
    str_acc = str_correct_count / total if total > 0 else 0
    valid_sql_acc = valid_sql_count / total if total > 0 else 0
    
    print(f"\n=== 评估结果 ===")
    print(f"Execution Accuracy: {exec_acc:.4f} ({exec_correct_count}/{total})")
    print(f"String-Match Accuracy: {str_acc:.4f} ({str_correct_count}/{total})")
    print(f"Valid SQL Accuracy: {valid_sql_acc:.4f} ({valid_sql_count}/{total}) (语法正确且可执行)")

    
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
    evaluate_spider(tokenizer, model, test_data, limit=100) 
