# AdaLoRA target_modules 选择指南
# 详细解释如何确定应该微调 Query, Key, Value 中的哪些模块

import torch
from transformers import BertModel, BertTokenizer, AutoModel
import numpy as np

class TargetModulesAnalyzer:
    """分析和选择最优target_modules的工具"""
    
    def __init__(self, model_name="bert-base-uncased"):
        self.model_name = model_name
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
    
    def print_all_linear_modules(self):
        """打印模型中所有的线性层，帮助确定target_modules"""
        print("="*70)
        print(f"模型 {self.model_name} 中所有的线性层:")
        print("="*70)
        
        linear_modules = {}
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                print(f"模块名: {name:<50} | 形状: {module.weight.shape}")
                linear_modules[name] = module.weight.shape
        
        return linear_modules
    
    def explain_qkv_roles(self):
        """解释Q, K, V在注意力机制中的作用"""
        print("\n" + "="*70)
        print("Q (Query), K (Key), V (Value) 在注意力机制中的作用")
        print("="*70)
        
        explanations = {
            "Query (查询)": {
                "作用": "表示当前token想要查找什么信息",
                "计算公式": "Q = X × W_q",
                "影响": "决定当前词关注哪些其他词",
                "微调效果": "改变模型的注意力焦点和查询模式",
                "适用场景": [
                    "需要改变模型关注重点的任务",
                    "情感分析（关注情感词）",
                    "关系抽取（关注实体关系）"
                ]
            },
            
            "Key (键)": {
                "作用": "表示每个token提供什么信息可以被查询",
                "计算公式": "K = X × W_k",
                "影响": "决定哪些词可以被其他词关注",
                "微调效果": "改变token的可发现性和匹配模式",
                "适用场景": [
                    "信息检索任务",
                    "文档匹配",
                    "问答系统（匹配问题和答案）"
                ]
            },
            
            "Value (值)": {
                "作用": "表示每个token实际传递的信息内容",
                "计算公式": "V = X × W_v",
                "影响": "决定注意力加权后传递什么信息",
                "微调效果": "改变信息的表示和传递方式",
                "适用场景": [
                    "需要改变语义表示的任务",
                    "文本分类",
                    "情感分析"
                ]
            }
        }
        
        for module_name, details in explanations.items():
            print(f"\n【{module_name}】")
            print(f"  作用: {details['作用']}")
            print(f"  计算: {details['计算公式']}")
            print(f"  影响: {details['影响']}")
            print(f"  微调效果: {details['微调效果']}")
            print(f"  适用场景:")
            for scenario in details['适用场景']:
                print(f"    - {scenario}")
    
    def attention_mechanism_demo(self):
        """演示注意力机制中Q, K, V的计算过程"""
        print("\n" + "="*70)
        print("注意力机制计算过程演示")
        print("="*70)
        
        print("""
        完整的注意力计算过程:
        
        1. 线性投影:
           Q = X × W_q  (查询矩阵)
           K = X × W_k  (键矩阵)
           V = X × W_v  (值矩阵)
        
        2. 计算注意力分数:
           Attention_Score = (Q × K^T) / √d_k
        
        3. Softmax归一化:
           Attention_Weight = Softmax(Attention_Score)
        
        4. 加权求和:
           Output = Attention_Weight × V
        
        关键洞察:
        - Q和K决定"注意力分布" (哪些词重要)
        - V决定"传递的内容" (传递什么信息)
        - 微调Q: 改变"我想找什么"
        - 微调K: 改变"我能被谁找到"
        - 微调V: 改变"我传递什么"
        """)

def get_module_selection_recommendations():
    """根据不同任务类型推荐target_modules配置"""
    
    print("\n" + "="*70)
    print("不同任务类型的 target_modules 推荐配置")
    print("="*70)
    
    recommendations = {
        "文本分类 (如IMDB情感分析)": {
            "推荐配置": ["query", "value"],
            "原因": [
                "Query: 帮助模型关注情感关键词",
                "Value: 调整情感信息的表示",
                "不需要Key: 键的匹配模式对分类影响较小"
            ],
            "参数量": "适中 (2个模块)",
            "性能": "高性价比",
            "代码示例": 'target_modules=["query", "value"]'
        },
        
        "命名实体识别 (NER)": {
            "推荐配置": ["query", "key", "value"],
            "原因": [
                "Query: 定位实体边界",
                "Key: 实体之间的关系匹配",
                "Value: 实体类型的语义表示"
            ],
            "参数量": "较多 (3个模块)",
            "性能": "最全面",
            "代码示例": 'target_modules=["query", "key", "value"]'
        },
        
        "问答系统": {
            "推荐配置": ["query", "key"],
            "原因": [
                "Query: 问题的查询表示",
                "Key: 答案段落的匹配特征",
                "Value保持不变: 利用预训练的语义知识"
            ],
            "参数量": "适中 (2个模块)",
            "性能": "专注匹配",
            "代码示例": 'target_modules=["query", "key"]'
        },
        
        "文本生成": {
            "推荐配置": ["query", "value", "dense"],
            "原因": [
                "Query: 控制生成关注点",
                "Value: 调整生成内容",
                "Dense: 前馈层也很重要"
            ],
            "参数量": "较多",
            "性能": "强表达力",
            "代码示例": 'target_modules=["query", "value", "intermediate", "output"]'
        },
        
        "资源受限场景": {
            "推荐配置": ["query"] or ["value"],
            "原因": [
                "只微调单个模块",
                "最小化参数量",
                "Query通常效果最好"
            ],
            "参数量": "最少 (1个模块)",
            "性能": "基础性能",
            "代码示例": 'target_modules=["query"]'
        },
        
        "极致性能场景": {
            "推荐配置": ["query", "key", "value", "output"],
            "原因": [
                "覆盖所有注意力组件",
                "包含输出投影层",
                "最大化适应能力"
            ],
            "参数量": "最多 (4+个模块)",
            "性能": "最高上限",
            "代码示例": 'target_modules=["query", "key", "value", "dense"]'
        }
    }
    
    for task_type, config in recommendations.items():
        print(f"\n【{task_type}】")
        print(f"  推荐配置: {config['推荐配置']}")
        print(f"  参数量: {config['参数量']}")
        print(f"  性能: {config['性能']}")
        print(f"  原因:")
        for reason in config['原因']:
            print(f"    - {reason}")
        print(f"  代码: {config['代码示例']}")

def analyze_parameter_impact():
    """分析不同target_modules配置对参数量的影响"""
    
    print("\n" + "="*70)
    print("target_modules 配置对参数量的影响分析")
    print("="*70)
    
    # 假设 BERT-base 配置: hidden_size=768, num_layers=12
    hidden_size = 768
    num_layers = 12
    
    def calculate_params(modules, init_r=16, target_r=8):
        """计算参数量"""
        params_per_layer = 0
        
        if "query" in modules:
            params_per_layer += init_r * hidden_size * 2  # A和B矩阵
        if "key" in modules:
            params_per_layer += init_r * hidden_size * 2
        if "value" in modules:
            params_per_layer += init_r * hidden_size * 2
        if "dense" in modules or "output" in modules:
            params_per_layer += init_r * hidden_size * 2
        
        total_params = params_per_layer * num_layers
        return total_params
    
    configurations = [
        (["query"], "最小配置"),
        (["query", "value"], "标准配置 (推荐)"),
        (["query", "key", "value"], "全注意力配置"),
        (["query", "value", "dense"], "注意力+前馈"),
        (["query", "key", "value", "dense"], "最大配置"),
    ]
    
    print(f"\n{'配置':<30} | {'模块数':<8} | {'参数量':<12} | {'相对比例'}")
    print("-" * 70)
    
    base_params = calculate_params(["query"])
    for modules, desc in configurations:
        params = calculate_params(modules)
        ratio = params / base_params
        print(f"{desc:<30} | {len(modules):<8} | {params:>10,} | {ratio:.2f}x")

def experimental_comparison():
    """实验对比不同配置的效果"""
    
    print("\n" + "="*70)
    print("实验对比: 不同 target_modules 配置的性能")
    print("="*70)
    
    print("""
    基于IMDB情感分析任务的实验结果 (示例数据):
    
    配置                    | 参数量  | 训练时间 | 准确率 | 性价比
    -----------------------|---------|---------|--------|--------
    ["query"]              | 295K    | 1.0x    | 83.2%  | ★★★☆☆
    ["value"]              | 295K    | 1.0x    | 82.8%  | ★★★☆☆
    ["query", "value"]     | 590K    | 1.2x    | 85.4%  | ★★★★★ (推荐)
    ["query", "key"]       | 590K    | 1.2x    | 84.1%  | ★★★★☆
    ["key", "value"]       | 590K    | 1.2x    | 83.9%  | ★★★☆☆
    ["query", "key", "value"] | 885K | 1.5x    | 86.2%  | ★★★★☆
    ["all attention"]      | 1.18M   | 2.0x    | 86.8%  | ★★★☆☆
    
    关键发现:
    1. Query是最重要的单一模块
    2. Query + Value 组合性价比最高
    3. 添加Key带来的提升有限
    4. 全量配置提升边际收益递减
    
    推荐策略:
    - 快速实验: ["query"]
    - 生产部署: ["query", "value"]
    - 追求极致: ["query", "key", "value"]
    """)

def show_bert_attention_structure():
    """展示BERT注意力结构，帮助理解模块命名"""
    
    print("\n" + "="*70)
    print("BERT 模型注意力结构详解")
    print("="*70)
    
    print("""
    BERT 注意力层的完整结构:
    
    BertAttention
    ├── self (BertSelfAttention)
    │   ├── query: Linear(768, 768)      ← target_modules: ["query"]
    │   ├── key: Linear(768, 768)        ← target_modules: ["key"]
    │   ├── value: Linear(768, 768)      ← target_modules: ["value"]
    │   └── dropout: Dropout(0.1)
    │
    └── output (BertSelfOutput)
        ├── dense: Linear(768, 768)       ← target_modules: ["dense"]
        ├── LayerNorm
        └── dropout
    
    BERT 前馈网络结构:
    
    BertIntermediate
    └── dense: Linear(768, 3072)          ← target_modules: ["intermediate"]
    
    BertOutput  
    └── dense: Linear(3072, 768)          ← target_modules: ["output"]
    
    在 AdaLoRA 中的命名映射:
    - "query" → encoder.layer.*.attention.self.query
    - "key" → encoder.layer.*.attention.self.key
    - "value" → encoder.layer.*.attention.self.value
    - "dense" → encoder.layer.*.attention.output.dense
    """)

def create_selection_decision_tree():
    """创建target_modules选择决策树"""
    
    print("\n" + "="*70)
    print("target_modules 选择决策树")
    print("="*70)
    
    print("""
    开始选择 target_modules
           |
           ├─ 参数预算是否极度受限？
           |  ├─ 是 → ["query"] (单模块)
           |  └─ 否 → 继续
           |
           ├─ 任务是否需要改变注意力焦点？
           |  ├─ 是 (情感分析、分类) → 包含 "query"
           |  └─ 否 → 继续
           |
           ├─ 任务是否需要匹配/检索？
           |  ├─ 是 (问答、搜索) → 包含 "key"
           |  └─ 否 → 继续
           |
           ├─ 任务是否需要改变语义表示？
           |  ├─ 是 (大多数任务) → 包含 "value"
           |  └─ 否 → 继续
           |
           ├─ 是否追求最高性能？
           |  ├─ 是 → ["query", "key", "value", "dense"]
           |  └─ 否 → ["query", "value"] (平衡选择)
           
    对于IMDB情感分析项目:
    ✓ 需要关注情感词 → 需要 Query
    ✓ 需要调整情感表示 → 需要 Value
    ✗ 不需要特殊匹配 → 不太需要 Key
    
    最终推荐: ["query", "value"]
    """)

def practical_testing_code():
    """提供实际测试不同配置的代码"""
    
    print("\n" + "="*70)
    print("实际测试代码: 比较不同 target_modules 配置")
    print("="*70)
    
    code = '''
# 测试不同target_modules配置的脚本
from peft import AdaLoraConfig, get_peft_model
from transformers import BertForSequenceClassification
import torch

def test_target_modules_config(target_modules_list):
    """测试不同的target_modules配置"""
    
    results = []
    model_base = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", 
        num_labels=2
    )
    
    for target_modules in target_modules_list:
        print(f"\\n测试配置: {target_modules}")
        
        # 创建AdaLoRA配置
        config = AdaLoraConfig(
            init_r=16,
            target_r=8,
            lora_alpha=32,
            target_modules=target_modules,
            task_type="SEQ_CLS"
        )
        
        # 应用到模型
        model = get_peft_model(model_base, config)
        
        # 统计参数
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  总参数: {total_params:,}")
        print(f"  参数比例: {100 * trainable_params / total_params:.4f}%")
        
        results.append({
            "modules": target_modules,
            "trainable": trainable_params,
            "ratio": trainable_params / total_params
        })
    
    return results

# 测试不同配置
configs_to_test = [
    ["query"],
    ["value"],
    ["query", "value"],
    ["query", "key"],
    ["query", "key", "value"],
]

results = test_target_modules_config(configs_to_test)

# 输出对比
print("\\n" + "="*60)
print("配置对比结果:")
for r in results:
    print(f"{str(r['modules']):<30} | 参数: {r['trainable']:>8,} | 比例: {r['ratio']*100:.3f}%")
    '''
    
    print(code)

if __name__ == "__main__":
    # 1. 打印模型结构
    analyzer = TargetModulesAnalyzer()
    analyzer.print_all_linear_modules()
    
    # 2. 解释Q, K, V的作用
    analyzer.explain_qkv_roles()
    
    # 3. 演示注意力机制
    analyzer.attention_mechanism_demo()
    
    # 4. 展示BERT结构
    show_bert_attention_structure()
    
    # 5. 不同任务的推荐配置
    get_module_selection_recommendations()
    
    # 6. 参数量影响分析
    analyze_parameter_impact()
    
    # 7. 实验对比
    experimental_comparison()
    
    # 8. 决策树
    create_selection_decision_tree()
    
    # 9. 实际测试代码
    practical_testing_code()
    
    print("\n" + "="*70)
    print("总结: 对于IMDB情感分析项目")
    print("="*70)
    print("""
    推荐配置: target_modules=["query", "value"]
    
    理由:
    1. Query: 帮助模型学习关注情感关键词 (如"好"、"坏"、"精彩"等)
    2. Value: 调整情感信息的语义表示
    3. 不需要Key: 情感分类不需要复杂的token匹配
    4. 性价比高: 只用2个模块就能达到很好的效果
    5. 参数适中: 约590K可训练参数
    
    如果资源更充足，可以尝试:
    target_modules=["query", "key", "value"]
    
    如果资源受限，可以降级为:
    target_modules=["query"]
    """)