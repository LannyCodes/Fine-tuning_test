import requests
import json

# vLLM 默认地址
OPENAI_API_KEY = "EMPTY"  # vLLM 默认不需要 key
OPENAI_API_BASE = "http://localhost:8000/v1"

def query_vllm(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    # 构造符合 OpenAI 格式的请求
    data = {
        "model": "/kaggle/working/qwen2_spider_merged", # 模型名称，对应启动参数 --model
        "messages": [
            {"role": "system", "content": "You are a helpful assistant who translates natural language queries into SQL commands for the Spider dataset."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 200,
        "top_p": 0.95
    }
    
    try:
        response = requests.post(f"{OPENAI_API_BASE}/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    test_prompt = "Show me the names of all students who have a GPA greater than 3.5."
    print(f"Prompt: {test_prompt}")
    print("-" * 30)
    
    sql = query_vllm(test_prompt)
    if sql:
        print("Generated SQL:")
        print(sql)
    else:
        print("Failed to get response.")
