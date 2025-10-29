# PEFT TaskType 完整指南和选择策略

from peft import TaskType
import inspect

def show_all_task_types():
    """显示PEFT库中所有可用的任务类型"""
    
    print("=== PEFT TaskType 完整列表 ===")
    
    # 获取TaskType枚举的所有成员
    task_types = {
        "SEQ_CLS": "序列分类 (Sequence Classification)",
        "SEQ_2_SEQ_LM": "序列到序列语言模型 (Sequence-to-Sequence Language Model)", 
        "CAUSAL_LM": "因果语言模型 (Causal Language Model)",
        "TOKEN_CLS": "词元分类 (Token Classification)",
        "QUESTION_ANS": "问答任务 (Question Answering)",
        "FEATURE_EXTRACTION": "特征提取 (Feature Extraction)",
        "DIFFUSION": "扩散模型 (Diffusion Model)",
        "SEMANTIC_SEGMENTATION": "语义分割 (Semantic Segmentation)",
        "IMAGE_CLASSIFICATION": "图像分类 (Image Classification)",
        "OBJECT_DETECTION": "目标检测 (Object Detection)"
    }
    
    for task_type, description in task_types.items():
        print(f"TaskType.{task_type:<20} - {description}")
    
    return task_types

def get_task_type_recommendations():
    """获取不同场景的TaskType推荐"""
    
    recommendations = {
        "文本分类任务": {
            "task_type": "TaskType.SEQ_CLS",
            "描述": "情感分析、新闻分类、垃圾邮件检测等",
            "模型输出": "分类logits",
            "适用模型": "BERT, RoBERTa, DistilBERT等编码器模型",
            "示例": ["情感分析(正面/负面)", "新闻分类", "意图识别"]
        },
        
        "命名实体识别": {
            "task_type": "TaskType.TOKEN_CLS", 
            "描述": "NER、词性标注、槽位填充等词元级分类",
            "模型输出": "每个token的分类logits",
            "适用模型": "BERT, RoBERTa等编码器模型",
            "示例": ["人名地名识别", "词性标注", "医学实体提取"]
        },
        
        "文本生成": {
            "task_type": "TaskType.CAUSAL_LM",
            "描述": "GPT风格的自回归文本生成",
            "模型输出": "下一个token的概率分布",
            "适用模型": "GPT-2, GPT-3, LLaMA等解码器模型", 
            "示例": ["对话生成", "文章续写", "代码生成"]
        },
        
        "序列到序列": {
            "task_type": "TaskType.SEQ_2_SEQ_LM",
            "描述": "编码器-解码器架构的序列转换",
            "模型输出": "目标序列",
            "适用模型": "T5, BART, mT5等encoder-decoder模型",
            "示例": ["机器翻译", "文本摘要", "语法纠错"]
        },
        
        "问答系统": {
            "task_type": "TaskType.QUESTION_ANS",
            "描述": "阅读理解、抽取式问答",
            "模型输出": "答案span的start/end位置",
            "适用模型": "BERT, RoBERTa等编码器模型",
            "示例": ["SQuAD问答", "机器阅读理解", "FAQ系统"]
        },
        
        "特征提取": {
            "task_type": "TaskType.FEATURE_EXTRACTION",
            "描述": "获取文本的向量表示",
            "模型输出": "文本embedding向量",
            "适用模型": "BERT, Sentence-BERT等",
            "示例": ["文本相似度", "语义搜索", "聚类分析"]
        }
    }
    
    return recommendations

def analyze_current_project():
    """分析当前项目应该使用的TaskType"""
    
    print("=== 当前项目任务分析 ===")
    
    project_analysis = {
        "数据集": "IMDB电影评论数据集",
        "任务目标": "判断电影评论的情感倾向（正面/负面）",
        "输入": "电影评论文本",
        "输出": "二分类结果（0=负面, 1=正面）",
        "模型架构": "DistilBERT + 分类头",
        "推荐TaskType": "TaskType.SEQ_CLS"
    }
    
    print("项目特征分析:")
    for key, value in project_analysis.items():
        print(f"  {key}: {value}")
    
    print("\n选择TaskType.SEQ_CLS的原因:")
    reasons = [
        "1. 这是一个文本分类任务（情感二分类）",
        "2. 输入是完整的文本序列",  
        "3. 输出是单个分类标签",
        "4. 使用BERT类编码器模型",
        "5. 不需要生成新文本，只需分类判断"
    ]
    
    for reason in reasons:
        print(f"  {reason}")
    
    return "TaskType.SEQ_CLS"

def task_type_decision_tree():
    """TaskType选择决策树"""
    
    decision_tree = """
    TaskType选择决策树:
    
    ┌─ 是文本任务吗？
    │  ├─ 是 → 继续
    │  └─ 否 → 检查图像/多模态任务
    │
    ├─ 需要生成新文本吗？
    │  ├─ 是 → 
    │  │  ├─ 自回归生成(GPT风格) → TaskType.CAUSAL_LM
    │  │  ├─ 序列转换(T5风格) → TaskType.SEQ_2_SEQ_LM  
    │  │  └─ 问答生成 → TaskType.QUESTION_ANS
    │  │
    │  └─ 否 → 继续
    │
    ├─ 需要对每个词分类吗？
    │  ├─ 是 → TaskType.TOKEN_CLS (NER, 词性标注)
    │  └─ 否 → 继续
    │
    ├─ 是整个文本的分类吗？
    │  ├─ 是 → TaskType.SEQ_CLS (情感分析, 主题分类)
    │  └─ 否 → 继续
    │
    ├─ 只需要文本向量表示吗？
    │  ├─ 是 → TaskType.FEATURE_EXTRACTION
    │  └─ 否 → 检查其他任务类型
    
    当前项目路径: 文本任务 → 不生成新文本 → 不对词分类 → 整个文本分类 → TaskType.SEQ_CLS ✓
    """
    
    print(decision_tree)

def show_task_type_impact():
    """展示TaskType对模型的影响"""
    
    print("=== TaskType对模型配置的影响 ===")
    
    impacts = {
        "SEQ_CLS": {
            "模型头": "分类头 (Linear + Softmax)",
            "损失函数": "CrossEntropyLoss", 
            "评估指标": "Accuracy, F1-score, Precision, Recall",
            "输出维度": "num_labels",
            "目标模块": "通常是attention层 (query, key, value)"
        },
        
        "TOKEN_CLS": {
            "模型头": "每个token的分类头",
            "损失函数": "CrossEntropyLoss (token级别)",
            "评估指标": "Token-level F1, Entity-level F1",
            "输出维度": "sequence_length × num_labels", 
            "目标模块": "注意力层和前馈层"
        },
        
        "CAUSAL_LM": {
            "模型头": "语言模型头 (LMHead)",
            "损失函数": "CrossEntropyLoss (下一词预测)",
            "评估指标": "Perplexity, BLEU",
            "输出维度": "vocab_size",
            "目标模块": "通常是所有Transformer层"
        }
    }
    
    for task_type, details in impacts.items():
        print(f"\n{task_type}:")
        for aspect, description in details.items():
            print(f"  {aspect}: {description}")

def validate_task_type_choice(model_type, task_description):
    """验证TaskType选择是否正确"""
    
    validation_rules = {
        "SEQ_CLS": {
            "compatible_models": ["bert", "roberta", "distilbert", "albert"],
            "task_patterns": ["分类", "sentiment", "classification", "categorization"],
            "output_format": "single_label_per_sequence"
        },
        "TOKEN_CLS": {
            "compatible_models": ["bert", "roberta", "distilbert"],
            "task_patterns": ["ner", "pos", "tagging", "token", "entity"],
            "output_format": "label_per_token"
        },
        "CAUSAL_LM": {
            "compatible_models": ["gpt", "llama", "mistral", "qwen"],
            "task_patterns": ["generation", "chat", "completion"],
            "output_format": "generated_sequence"
        }
    }
    
    print(f"=== 验证TaskType选择 ===")
    print(f"模型类型: {model_type}")
    print(f"任务描述: {task_description}")
    
    # 当前项目验证
    current_model = "distilbert"
    current_task = "情感分类 sentiment classification"
    
    for task_type, rules in validation_rules.items():
        model_match = any(m in current_model.lower() for m in rules["compatible_models"])
        task_match = any(p in current_task.lower() for p in rules["task_patterns"])
        
        if model_match and task_match:
            print(f"\n✅ 推荐使用: TaskType.{task_type}")
            print(f"   模型兼容: {model_match}")
            print(f"   任务匹配: {task_match}")
            return f"TaskType.{task_type}"
    
    print("\n❌ 未找到完全匹配的TaskType，请手动确认")
    return None

if __name__ == "__main__":
    # 1. 显示所有可用的TaskType
    available_types = show_all_task_types()
    
    print("\n" + "="*60 + "\n")
    
    # 2. 显示TaskType推荐
    recommendations = get_task_type_recommendations()
    print("=== 不同场景的TaskType推荐 ===")
    for scenario, info in recommendations.items():
        print(f"\n【{scenario}】")
        print(f"  TaskType: {info['task_type']}")
        print(f"  描述: {info['描述']}")
        print(f"  适用模型: {info['适用模型']}")
        print(f"  示例: {', '.join(info['示例'])}")
    
    print("\n" + "="*60 + "\n")
    
    # 3. 分析当前项目
    recommended_task_type = analyze_current_project()
    
    print("\n" + "="*60 + "\n")
    
    # 4. 显示决策树
    task_type_decision_tree()
    
    print("\n" + "="*60 + "\n")
    
    # 5. 显示TaskType的影响
    show_task_type_impact()
    
    print("\n" + "="*60 + "\n")
    
    # 6. 验证选择
    validated_choice = validate_task_type_choice("distilbert", "sentiment classification")
    
    print(f"\n=== 最终推荐 ===")
    print(f"当前IMDB情感分析项目应该使用: {recommended_task_type}")
    print(f"验证结果: {validated_choice}")