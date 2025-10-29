import pandas as pd
from datasets import load_dataset

# 加载IMDB数据集
dataset = load_dataset("imdb")

# 提取正样本和负样本各1000条
pos_samples = [text for text, label in zip(dataset["train"]["text"], dataset["train"]["label"]) if label == 1][:1000]
neg_samples = [text for text, label in zip(dataset["train"]["text"], dataset["train"]["label"]) if label == 0][:1000]

# 创建DataFrame并添加标签
pos_df = pd.DataFrame({
    'text': pos_samples,
    'label': 'positive'
})

neg_df = pd.DataFrame({
    'text': neg_samples,
    'label': 'negative'
})

# 合并并打乱数据
combined_df = pd.concat([pos_df, neg_df], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存到本地文件
file_path = "imdb_samples_2000.csv"
combined_df.to_csv(file_path, index=False, encoding='utf-8')

print(f"已成功保存 {len(combined_df)} 条样本到 {file_path}")
print(f"其中正面样本: {len(pos_df)}, 负面样本: {len(neg_df)}")