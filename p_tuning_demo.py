import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from torch.optim import AdamW
from datasets import load_dataset
import random
import numpy as np
from tqdm import tqdm

# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# --------------------------
# 1. P-tuning模型定义
# --------------------------
class PTuningModel(nn.Module):
    def __init__(self, bert, tokenizer, prompt_len=10):
        super().__init__()
        self.bert = bert
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        
        # 可训练的软提示
        self.soft_prompt = nn.Parameter(
            torch.randn(prompt_len, bert.config.hidden_size) * 0.01
        )
        
        # 添加Dropout正则化
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(bert.config.hidden_size, 2)
        
        # 冻结BERT所有参数
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        
        # 扩展软提示到批次大小
        soft_prompt = self.soft_prompt.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 获取文本嵌入
        text_embeds = self.bert.embeddings(input_ids)
        
        # 拼接软提示和文本嵌入
        combined_embeds = torch.cat([soft_prompt, text_embeds], dim=1)
        
        # 调整注意力掩码
        prompt_mask = torch.ones(batch_size, self.prompt_len, device=input_ids.device)
        combined_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        
        # 通过BERT模型
        outputs = self.bert(inputs_embeds=combined_embeds, attention_mask=combined_mask)
        
        # 获取[CLS]向量并应用Dropout
        cls_embeds = outputs.last_hidden_state[:, 0, :]
        cls_embeds = self.dropout(cls_embeds)
        
        # 通过分类器
        logits = self.classifier(cls_embeds)
        return logits

# --------------------------
# 2. 数据集定义
# --------------------------
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.texts)

# --------------------------
# 3. 改进的数据增强
# --------------------------
def augment_text(text, tokenizer, p=0.1):
    tokens = tokenizer.tokenize(text)
    augmented_tokens = []
    
    for token in tokens:
        # 避免替换情感关键词
        if random.random() < p and token not in tokenizer.all_special_tokens and \
           not token.startswith("##") and len(token) > 2:
            random_token = random.choice(list(tokenizer.vocab.keys()))
            # 避免替换情感关键词（简化版）
            if not any(word in random_token for word in ["good", "bad", "terrible", "wonderful", "awful"]):
                augmented_tokens.append(random_token)
            else:
                augmented_tokens.append(token)
        else:
            augmented_tokens.append(token)
    
    return tokenizer.convert_tokens_to_string(augmented_tokens)

# --------------------------
# 4. 训练和评估函数
# --------------------------
def train(model, train_loader, optimizer, device, class_weights=None):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        logits = model(input_ids, attention_mask)
        
        # 使用带权重的交叉熵损失
        if class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=class_weights.to(device))
        else:
            loss = F.cross_entropy(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    acc = correct / total
    
    # 计算各类别准确率
    pos_correct = sum([p == l for p, l in zip(all_preds, all_labels) if l == 1])
    pos_total = sum(all_labels)
    neg_correct = sum([p == l for p, l in zip(all_preds, all_labels) if l == 0])
    neg_total = len(all_labels) - pos_total
    
    print(f"测试集准确率：{acc:.4f}")
    print(f"正面评论准确率：{pos_correct/pos_total:.4f} ({pos_correct}/{pos_total})")
    print(f"负面评论准确率：{neg_correct/neg_total:.4f} ({neg_correct}/{neg_total})")
    
    # 计算混淆矩阵
    tp = sum([p == l == 1 for p, l in zip(all_preds, all_labels)])
    tn = sum([p == l == 0 for p, l in zip(all_preds, all_labels)])
    fp = sum([p == 1 and l == 0 for p, l in zip(all_preds, all_labels)])
    fn = sum([p == 0 and l == 1 for p, l in zip(all_preds, all_labels)])
    
    print("\n混淆矩阵:")
    print(f"        预测正  预测负")
    print(f"实际正  {tp:5d}  {fn:5d}")
    print(f"实际负  {fp:5d}  {tn:5d}")
    
    return acc

# --------------------------
# 5. 主函数
# --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    
    # 加载IMDB数据集
    print("加载IMDB数据集...")
    dataset = load_dataset("imdb")
    
    # --------------------------
    # 关键修复：强制选择平衡的正负样本
    # --------------------------
    # 从训练集中分别提取1000条正面和1000条负面样本（共2000条）
    train_data = dataset["train"]
    
    # 筛选正面样本（label=1）和负面样本（label=0）
    pos_indices = [i for i, label in enumerate(train_data["label"]) if label == 1][:1000]
    neg_indices = [i for i, label in enumerate(train_data["label"]) if label == 0][:1000]
    
    # 合并并打乱
    all_indices = pos_indices + neg_indices
    random.shuffle(all_indices)
    
    # 提取平衡的训练数据
    train_texts = [train_data["text"][i] for i in all_indices]
    train_labels = [train_data["label"][i] for i in all_indices]
    
    # 测试集使用完整数据
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]
    
    # 验证数据平衡
    pos_count = sum(1 for l in train_labels if l == 1)
    neg_count = len(train_labels) - pos_count
    print(f"训练数据平衡检查：正面样本={pos_count}, 负面样本={neg_count}")
    if pos_count == 0 or neg_count == 0:
        print("严重错误：训练数据仍缺少类别，请检查数据集！")
        return
    
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # 数据增强（分别对正负面样本增强）
    print("应用数据增强...")
    pos_texts = [text for text, label in zip(train_texts, train_labels) if label == 1]
    neg_texts = [text for text, label in zip(train_texts, train_labels) if label == 0]
    
    augmented_pos = [augment_text(text, tokenizer) for text in pos_texts]
    augmented_neg = [augment_text(text, tokenizer) for text in neg_texts]
    
    # 合并增强数据（保持平衡）
    train_texts = pos_texts + neg_texts + augmented_pos + augmented_neg
    train_labels = [1]*len(pos_texts) + [0]*len(neg_texts) + [1]*len(augmented_pos) + [0]*len(augmented_neg)
    
    # 打乱数据
    combined = list(zip(train_texts, train_labels))
    random.shuffle(combined)
    train_texts, train_labels = zip(*combined)
    
    # 后续代码（数据集、模型、训练等）保持不变...
    
    # 创建数据集和数据加载器
    train_dataset = IMDBDataset(train_texts, train_labels, tokenizer)
    test_dataset = IMDBDataset(test_texts, test_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 初始化模型
    print("初始化P-tuning模型...")
    bert = BertModel.from_pretrained("bert-base-uncased").to(device)
    model = PTuningModel(bert, tokenizer, prompt_len=10).to(device)
    
    # 计算类别权重（添加安全检查）
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    total = len(train_labels)
    
    # 安全检查：确保不会除零
    if pos_count == 0 or neg_count == 0:
        print("警告：训练数据中存在类别缺失，无法计算类别权重")
        class_weights = None
    else:
        class_weights = torch.tensor([total/(2*neg_count), total/(2*pos_count)], dtype=torch.float)
        print(f"类别权重: 正面={class_weights[1]:.4f}, 负面={class_weights[0]:.4f}")
    
    # 优化器
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-5,
        weight_decay=0.01
    )
    
    # 早停机制
    best_acc = 0
    patience = 3
    counter = 0
    epochs = 10
    
    print("开始训练...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss = train(model, train_loader, optimizer, device, class_weights)
        print(f"训练损失: {train_loss:.4f}")
        
        val_acc = evaluate(model, test_loader, device)
        
        # 早停检查
        if val_acc > best_acc:
            best_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), "best_model.pt")
            print(f"模型保存 - 准确率: {val_acc:.4f}")
        else:
            counter += 1
            print(f"早停计数器: {counter}/{patience}")
            if counter >= patience:
                print(f"早停：{patience}个epoch准确率未提升")
                break
    
    # 加载最佳模型进行最终测试
    print("\n加载最佳模型进行最终测试...")
    model.load_state_dict(torch.load("best_model.pt"))
    evaluate(model, test_loader, device)
    
    # 测试特定评论
    test_comments = [
        "This movie is absolutely wonderful! The acting was great and the plot was engaging.",
        "Terrible! Worst movie I've ever seen. Don't waste your time.",
        "A must-see film that explores complex themes with incredible depth.",
        "The plot was predictable and the characters were flat. Save your money."
    ]
    
    print("\n测试特定评论:")
    model.eval()
    with torch.no_grad():
        for comment in test_comments:
            encoding = tokenizer(
                comment, 
                return_tensors="pt", 
                max_length=128, 
                padding="max_length", 
                truncation=True
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            
            logits = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=1)
            pred = "positive" if probs[0][1] > 0.5 else "negative"
            confidence = probs[0][1].item() if pred == "positive" else probs[0][0].item()
            
            print(f"\n评论: {comment}")
            print(f"预测: {pred} (置信度: {confidence:.4f})")

if __name__ == "__main__":
    main()