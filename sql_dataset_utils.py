import json

def format_schema(db_schema):
    """
    将数据库 Schema 格式化为字符串，放入 Prompt 中。
    Spider 数据集的 tables.json 通常包含 table_names 和 column_names。
    这里假设输入是一个简化的字典格式，实际使用需根据 Spider json 结构调整。
    """
    schema_str = ""
    for table_name, columns in db_schema.items():
        schema_str += f"Table: {table_name}\nColumns: {', '.join(columns)}\n\n"
    return schema_str.strip()

def process_spider_data(example, tokenizer, max_length=1024):
    """
    处理 Spider/BIRD 格式的数据
    example: 包含 'question', 'query' (sql), 'db_schema' 的字典
    """
    
    # 1. 构建包含 Schema 的 Prompt
    # schema_str = format_schema(example['db_schema']) 
    # 注意：实际 Spider 数据集需要根据 db_id 从 tables.json 查找对应的 schema
    # 这里为了演示，假设 example 中直接包含了格式化好的 schema_str
    
    schema_context = example.get('schema_context', "") 
    question = example['question']
    sql_label = example['query']
    
    prompt = (
        f"Human: 请根据给定的数据库 Schema 编写 SQL 语句。\n\n"
        f"{schema_context}\n\n"
        f"问题: {question}\n"
        f"Assistant: "
    )
    
    # 2. 编码
    # SQL 任务通常 Prompt 很长，Output 较短
    inputs = tokenizer(
        prompt + sql_label + tokenizer.eos_token,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    
    inputs["labels"] = inputs["input_ids"].copy()
    
    # 将 Prompt 部分的 label 设为 -100，只计算 SQL 输出部分的 Loss
    # 这一步对于长 Prompt 的任务非常重要，能加速收敛
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(prompt_ids)
    
    # 只要 prompt_len 小于 max_length，就把前面的 label 掩盖掉
    if prompt_len < max_length:
        inputs["labels"][:prompt_len] = [-100] * prompt_len
        
    return inputs

# ==========================================
# 模拟数据演示
# ==========================================
if __name__ == "__main__":
    # 模拟一条 Spider 格式的数据
    sample_data = {
        "question": "列出所有年龄大于 20 岁的学生姓名。",
        "query": "SELECT name FROM students WHERE age > 20;",
        "schema_context": "Table: students\nColumns: id, name, age, gender\n\nTable: courses\nColumns: course_id, title, credits"
    }
    
    print("--- Prompt 预览 ---")
    print(f"Human: 请根据给定的数据库 Schema 编写 SQL 语句。\n\n{sample_data['schema_context']}\n\n问题: {sample_data['question']}\nAssistant: {sample_data['query']}")
