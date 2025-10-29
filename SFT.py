import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# 设置环境
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 增强国内镜像源配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 禁用SSL证书验证（解决连接关闭问题）
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载中文预训练模型和分词器
model_name = "uer/gpt2-chinese-cluecorpussmall"  # 中文GPT-2模型
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 添加缺失的填充标记
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(model_name)
# 调整模型嵌入层以适应新添加的标记
model.resize_token_embeddings(len(tokenizer))
model = model.to(device)

# 2. 准备指令数据集（100样本扩展版）
instructions = [
    {
        "instruction": "什么是光合作用？",
        "output": "光合作用是植物利用阳光、水和二氧化碳制造氧气和糖分的过程。这个过程发生在叶绿体中，是地球上大多数生命的能量来源。"
    },
    {
        "instruction": "解释Python中的列表推导式",
        "output": "列表推导式是Python中创建列表的简洁方法，语法为：[表达式 for 变量 in 可迭代对象 if 条件]。例如：[x*2 for x in range(5) if x%2==0] 会生成 [0, 4, 8]。"
    },
    {
        "instruction": "简述太阳系的结构",
        "output": "太阳系由太阳、八大行星、卫星、小行星和彗星组成。行星按离太阳距离依次为水星、金星、地球、火星、木星、土星、天王星、海王星。其中木星是体积最大的行星。"
    },
    {
        "instruction": "什么是人工智能？",
        "output": "人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。它包括机器学习、自然语言处理、计算机视觉等领域。"
    },
    {
        "instruction": "解释区块链技术",
        "output": "区块链是一种分布式账本技术，通过加密和去中心化方式存储数据。每个数据块包含前一个块的哈希值，形成不可篡改的链式结构。"
    },
    {
        "instruction": "如何解决Python中的内存泄漏问题？",
        "output": "解决Python内存泄漏可采取以下步骤：1.使用memory_profiler定位泄漏源；2.检查全局变量和缓存；3.确保正确关闭文件和网络连接；4.使用弱引用处理循环引用。"
    },
    {
        "instruction": "什么是机器学习？",
        "output": "机器学习是人工智能的一个分支，研究如何让计算机在没有明确编程的情况下学习。主要分为监督学习、无监督学习和强化学习三大类。"
    },
    {
        "instruction": "解释什么是HTTP协议",
        "output": "HTTP（超文本传输协议）是用于传输超媒体文档（如HTML）的应用层协议。它基于客户端-服务器模型，使用请求-响应机制进行通信。"
    },
    {
        "instruction": "如何实现Python中的多线程？",
        "output": "Python中实现多线程可使用threading模块：import threading; t = threading.Thread(target=func); t.start(); t.join()。注意GIL限制导致CPU密集型任务难以真正并行。"
    },
    {
        "instruction": "什么是数据库索引？",
        "output": "数据库索引是一种数据结构，用于快速查询数据库表中的特定记录。常见类型包括B树索引、哈希索引和全文索引，适当的索引可大幅提高查询性能。"
    },
    {
        "instruction": "解释面向对象编程的三大特性",
        "output": "面向对象编程三大特性：1.封装：将数据和方法捆绑在一起；2.继承：子类继承父类的属性和方法；3.多态：不同对象对同一消息作出不同响应。"
    },
    {
        "instruction": "什么是云计算？",
        "output": "云计算是通过互联网提供计算资源（服务器、存储、数据库等）的服务模式，包括IaaS（基础设施即服务）、PaaS（平台即服务）和SaaS（软件即服务）。"
    },
    {
        "instruction": "如何在Python中读取CSV文件？",
        "output": "Python读取CSV文件可使用csv模块或pandas库：import pandas as pd; df = pd.read_csv('file.csv')。pandas提供更强大的数据处理功能。"
    },
    {
        "instruction": "解释TCP/IP协议",
        "output": "TCP/IP是一组网络通信协议，包括TCP（传输控制协议）和IP（网际协议）。TCP负责数据传输的可靠性，IP负责数据包的路由选择。"
    },
    {
        "instruction": "什么是深度学习？",
        "output": "深度学习是机器学习的子集，使用多层神经网络处理数据。它能自动提取特征，在图像识别、自然语言处理等领域取得突破性成果。"
    },
    {
        "instruction": "如何防止SQL注入攻击？",
        "output": "防止SQL注入的方法：1.使用参数化查询；2.输入验证和过滤；3.使用ORM框架；4.限制数据库权限；5.避免动态SQL拼接。"
    },
    {
        "instruction": "解释RESTful API设计原则",
        "output": "RESTful API设计原则：1.使用HTTP方法表达操作（GET/POST/PUT/DELETE）；2.资源为中心的URL设计；3.无状态通信；4.支持JSON/XML格式；5.提供超媒体链接。"
    },
    {
        "instruction": "Python中的装饰器有什么作用？",
        "output": "装饰器是修改函数或类行为的函数，用于代码重用、日志记录、性能测试、权限验证等场景。使用@符号语法，本质是高阶函数。"
    },
    {
        "instruction": "什么是容器化技术？",
        "output": "容器化技术（如Docker）将应用及其依赖打包成标准化单元，确保在不同环境中一致运行。相比虚拟机更轻量，启动更快，资源占用更少。"
    },
    {
        "instruction": "如何实现Python中的异常处理？",
        "output": "Python异常处理使用try-except结构：try: risky_operation() except ExceptionType as e: handle_error(e) finally: cleanup_resources()。可捕获并处理特定类型的异常。"
    },
    {
        "instruction": "解释什么是Git版本控制",
        "output": "Git是分布式版本控制系统，用于跟踪文件变化、协作开发和代码管理。核心概念包括仓库、提交、分支、合并和远程仓库。"
    },
    {
        "instruction": "什么是自然语言处理？",
        "output": "自然语言处理（NLP）是AI的一个分支，研究计算机理解、解释和生成人类语言的能力。应用包括机器翻译、情感分析、聊天机器人等。"
    },
    {
        "instruction": "如何优化Python代码性能？",
        "output": "Python性能优化方法：1.使用内置函数和库；2.避免全局变量；3.使用生成器节省内存；4.利用多线程/多进程；5.关键部分用C扩展。"
    },
    {
        "instruction": "解释什么是微服务架构",
        "output": "微服务架构将应用拆分为小型、自治的服务，每个服务运行在独立进程中，通过API通信。优点包括可扩展性、技术多样性和团队自治。"
    },
    {
        "instruction": "Python中的生成器有什么用途？",
        "output": "生成器通过yield关键字创建，用于生成迭代器。优点是延迟计算、节省内存，适用于处理大数据集或无限序列。语法：def generator(): yield value。"
    },
    {
        "instruction": "什么是计算机网络？",
        "output": "计算机网络是互连的计算设备集合，通过通信链路交换数据。按规模分为LAN（局域网）、WAN（广域网）和MAN（城域网）等类型。"
    },
    {
        "instruction": "如何在Python中操作JSON数据？",
        "output": "Python使用json模块处理JSON：import json; data = json.loads(json_str); json_str = json.dumps(data)。支持基本数据类型的序列化和反序列化。"
    },
    {
        "instruction": "解释什么是机器学习中的过拟合",
        "output": "过拟合指模型在训练数据上表现很好，但在新数据上泛化能力差。解决方法：增加数据量、正则化、早停、简化模型结构、交叉验证。"
    },
    {
        "instruction": "什么是敏捷开发？",
        "output": "敏捷开发是迭代式软件开发方法，强调适应性规划、快速交付和持续改进。常见框架包括Scrum、Kanban和Extreme Programming (XP)。"
    },
    {
        "instruction": "Python中的上下文管理器有什么作用？",
        "output": "上下文管理器通过with语句管理资源，确保资源正确获取和释放。用于文件操作、数据库连接等场景，语法：with open('file.txt') as f: ...。"
    },
    {
        "instruction": "解释什么是区块链的共识机制",
        "output": "共识机制是区块链网络中节点达成一致的算法，确保数据一致性。常见机制包括工作量证明（PoW）、权益证明（PoS）和委托权益证明（DPoS）。"
    },
    {
        "instruction": "如何在Python中实现单例模式？",
        "output": "Python单例模式实现方法：1.使用模块级变量；2.重写__new__方法；3.使用装饰器；4.使用元类。推荐使用模块方式，简单且线程安全。"
    },
    {
        "instruction": "什么是数据结构？",
        "output": "数据结构是计算机中组织和存储数据的特定方式，包括数组、链表、栈、队列、树、图等。选择合适的数据结构可提高算法效率。"
    },
    {
        "instruction": "解释什么是API网关",
        "output": "API网关是客户端与微服务之间的中间层，提供路由、认证、限流、监控等功能。简化客户端调用，隐藏服务复杂性，常见实现有Kong、Zuul。"
    },
    {
        "instruction": "Python中的多进程和多线程有什么区别？",
        "output": "多线程共享内存空间，受GIL限制；多进程拥有独立内存空间，可真正并行。CPU密集型任务适合多进程，I/O密集型适合多线程。"
    },
    {
        "instruction": "什么是大数据？",
        "output": "大数据指规模超出传统工具处理能力的数据集，具有4V特征：Volume（容量）、Velocity（速度）、Variety（多样性）和Value（价值）。"
    },
    {
        "instruction": "如何使用Python进行单元测试？",
        "output": "Python单元测试使用unittest模块：import unittest; class TestMyFunc(unittest.TestCase): def test_case(self): self.assertEqual(func(), expected)。运行：unittest.main()。"
    },
    {
        "instruction": "解释什么是CI/CD",
        "output": "CI/CD指持续集成/持续部署，是DevOps实践。CI自动构建和测试代码；CD自动将通过测试的代码部署到生产环境，提高开发效率和质量。"
    },
    {
        "instruction": "Python中的列表和元组有什么区别？",
        "output": "列表（list）是可变序列，支持增删改操作；元组（tuple）是不可变序列，创建后不能修改。元组比列表更轻量，可作为字典键使用。"
    },
    {
        "instruction": "什么是计算机视觉？",
        "output": "计算机视觉是AI的一个分支，研究如何让计算机理解图像内容。应用包括图像识别、目标检测、人脸识别、自动驾驶等领域。"
    },
    {
        "instruction": "如何在Python中连接MySQL数据库？",
        "output": "Python连接MySQL可使用mysql-connector或pymysql库：import pymysql; conn = pymysql.connect(host='localhost', user='user', password='pass', db='db'); cursor = conn.cursor()。"
    },
    {
        "instruction": "解释什么是缓存机制",
        "output": "缓存机制将频繁访问的数据存储在快速存储介质中，减少对慢速存储的访问。常见缓存策略：LRU（最近最少使用）、FIFO（先进先出）、LFU（最不经常使用）。"
    },
    {
        "instruction": "Python中的装饰器如何实现带参数功能？",
        "output": "带参数装饰器需要三层嵌套：def decorator(param): def wrapper(func): def inner(*args, **kwargs): return func(*args, **kwargs); return inner; return wrapper。使用：@decorator(param)。"
    },
    {
        "instruction": "什么是分布式系统？",
        "output": "分布式系统是由多个独立计算机通过网络协同工作的系统，表现为单一系统。特点包括资源共享、并发性、可扩展性和容错性。"
    },
    {
        "instruction": "如何在Python中处理日期和时间？",
        "output": "Python处理日期时间使用datetime模块：from datetime import datetime; now = datetime.now(); formatted = now.strftime('%Y-%m-%d %H:%M:%S')。pandas库提供更高级的时间序列功能。"
    },
    {
        "instruction": "解释什么是ORM框架",
        "output": "ORM（对象关系映射）将数据库表映射为对象，允许使用面向对象方式操作数据库。优点：简化SQL操作、提高开发效率、数据库无关性。Python中常见ORM有SQLAlchemy、Django ORM。"
    },
    {
        "instruction": "Python中的函数式编程有哪些特点？",
        "output": "Python函数式编程特点：1.使用纯函数（无副作用）；2.避免可变状态；3.使用高阶函数（map/filter/reduce）；4.支持匿名函数lambda；5.函数可作为参数和返回值。"
    },
    {
        "instruction": "什么是边缘计算？",
        "output": "边缘计算将数据处理放在网络边缘（靠近数据源的设备），而非集中式云服务器。优点：低延迟、节省带宽、提高隐私性，适用于IoT和实时应用。"
    },
    {
        "instruction": "如何在Python中实现异步编程？",
        "output": "Python异步编程使用asyncio库：async def func(): await coroutine(); loop = asyncio.get_event_loop(); loop.run_until_complete(func())。关键字：async/await，适用于I/O密集型任务。"
    },
    {
        "instruction": "解释什么是机器学习中的梯度下降",
        "output": "梯度下降是优化算法，通过沿损失函数梯度（最快下降方向）迭代调整参数，最小化损失函数。变种包括批量梯度下降、随机梯度下降和小批量梯度下降。"
    },
    {
        "instruction": "什么是DevOps？",
        "output": "DevOps是开发（Development）和运维（Operations）的结合，通过自动化和协作提高软件交付速度和质量。核心实践包括CI/CD、监控和日志管理。"
    },
    {
        "instruction": "Python中的模块和包有什么区别？",
        "output": "模块是单个.py文件，包是包含多个模块的目录（需有__init__.py）。模块用于组织函数和类，包用于组织相关模块，形成命名空间。"
    },
    {
        "instruction": "什么是强化学习？",
        "output": "强化学习是机器学习的一种，智能体通过与环境交互，从奖励中学习最优行为策略。应用包括游戏AI、机器人控制和资源管理。"
    },
    {
        "instruction": "如何在Python中进行文件操作？",
        "output": "Python文件操作使用open()函数：with open('file.txt', 'r') as f: content = f.read()。模式包括'r'(读)、'w'(写)、'a'(追加)、'b'(二进制)等，with语句自动关闭文件。"
    },
    {
        "instruction": "解释什么是负载均衡",
        "output": "负载均衡将网络流量分配到多个服务器，提高系统可用性和性能。方法包括轮询、加权轮询、IP哈希和最少连接。常见实现有Nginx、HAProxy。"
    },
    {
        "instruction": "Python中的迭代器和可迭代对象有什么区别？",
        "output": "可迭代对象（如列表）实现__iter__()方法，返回迭代器；迭代器实现__next__()方法，逐个返回元素。迭代器只能遍历一次，使用next()或for循环访问。"
    },
    {
        "instruction": "什么是量子计算？",
        "output": "量子计算利用量子力学原理（叠加态、纠缠）进行计算，可解决传统计算机难以处理的问题。量子比特可同时表示0和1，潜在计算能力远超经典计算机。"
    },
    {
        "instruction": "如何使用Python进行数据可视化？",
        "output": "Python数据可视化库：1.matplotlib：基础绘图库；2.seaborn：统计可视化；3.plotly：交互式可视化；4.pandas：基于matplotlib的简化接口。示例：df.plot(kind='bar')。"
    },
    {
        "instruction": "解释什么是容器编排",
        "output": "容器编排管理容器的生命周期，包括部署、扩展、网络和存储。Kubernetes是主流容器编排平台，提供自动扩缩容、自愈能力和滚动更新等功能。"
    },
    {
        "instruction": "Python中的类方法和静态方法有什么区别？",
        "output": "类方法（@classmethod）接收类作为第一个参数(cls)，可访问类属性；静态方法（@staticmethod）无特殊参数，不能访问类或实例属性。都可通过类名调用。"
    },
    {
        "instruction": "什么是数据挖掘？",
        "output": "数据挖掘从大量数据中提取隐含、有用的信息和知识。常用技术包括分类、聚类、关联规则挖掘和异常检测，应用于市场营销、欺诈检测等领域。"
    },
    {
        "instruction": "如何在Python中实现HTTP请求？",
        "output": "Python发送HTTP请求使用requests库：import requests; response = requests.get(url); print(response.json())。支持GET/POST/PUT/DELETE等方法，自动处理JSON。"
    },
    {
        "instruction": "解释什么是面向切面编程",
        "output": "AOP（面向切面编程）将横切关注点（如日志、事务）与业务逻辑分离，通过切面模块化。Python可通过装饰器或第三方库（如aspectlib）实现AOP。"
    },
    {
        "instruction": "Python中的垃圾回收机制是什么？",
        "output": "Python使用引用计数为主、分代回收为辅的垃圾回收机制。引用计数为0时释放内存，分代回收处理循环引用。可通过gc模块控制垃圾回收行为。"
    },
    {
        "instruction": "什么是物联网（IoT）？",
        "output": "物联网是物理设备通过网络互连的系统，设备嵌入传感器和软件，实现数据收集和交换。应用包括智能家居、工业监控和智能城市。"
    },
    {
        "instruction": "如何在Python中解析命令行参数？",
        "output": "Python解析命令行参数使用argparse模块：import argparse; parser = argparse.ArgumentParser(); parser.add_argument('--name'); args = parser.parse_args()。支持位置参数和可选参数。"
    },
    {
        "instruction": "解释什么是机器学习中的特征工程",
        "output": "特征工程是将原始数据转换为模型可用特征的过程，包括特征选择、提取、转换和缩放。良好的特征工程可显著提高模型性能，是机器学习的关键步骤。"
    },
    {
        "instruction": "Python中的深拷贝和浅拷贝有什么区别？",
        "output": "浅拷贝（copy.copy()）创建新对象，但引用原对象中的子对象；深拷贝（copy.deepcopy()）创建新对象并递归拷贝所有子对象。对不可变对象，两者效果相同。"
    },
    {
        "instruction": "什么是网络安全？",
        "output": "网络安全保护网络和系统免受未经授权的访问、使用、泄露、破坏。关键领域包括加密、防火墙、入侵检测、漏洞评估和安全审计。"
    },
    {
        "instruction": "如何在Python中生成随机数？",
        "output": "Python生成随机数使用random模块：import random; random_num = random.randint(1, 100); random_list = random.sample(range(10), 5)。需注意伪随机数的种子设置。"
    },
    {
        "instruction": "解释什么是微前端架构",
        "output": "微前端将前端应用拆分为小型、自治的应用，每个应用可独立开发、测试和部署。通过容器应用组合，实现技术栈无关和团队自治，类似前端的微服务。"
    },
    {
        "instruction": "Python中的闭包有什么用途？",
        "output": "闭包是嵌套函数，内部函数引用外部函数的变量。用途：1.数据隐藏和封装；2.创建工厂函数；3.实现装饰器；4.延迟计算。需注意变量作用域和生命周期。"
    },
    {
        "instruction": "什么是云计算的服务模式？",
        "output": "云计算服务模式：1.IaaS（基础设施即服务）：提供虚拟机、存储等基础设施；2.PaaS（平台即服务）：提供开发和部署平台；3.SaaS（软件即服务）：提供即用软件应用。"
    },
    {
        "instruction": "如何在Python中实现日志记录？",
        "output": "Python日志记录使用logging模块：import logging; logging.basicConfig(level=logging.INFO); logging.info('message')。支持不同级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）和输出目标。"
    },
    {
        "instruction": "解释什么是机器学习模型的评估指标",
        "output": "分类评估指标：准确率、精确率、召回率、F1分数、ROC/AUC；回归评估指标：MAE、MSE、RMSE、R²；聚类评估指标：轮廓系数、兰德指数。选择指标需根据任务类型。"
    },
    {
        "instruction": "Python中的__init__和__new__方法有什么区别？",
        "output": "__new__是创建对象的方法，返回实例；__init__是初始化对象的方法，无返回值。__new__在__init__之前调用，可用于实现单例模式或控制实例创建过程。"
    },
    {
        "instruction": "什么是5G技术？",
        "output": "5G是第五代移动通信技术，相比4G提供更高带宽、更低延迟和更多连接。关键特性：eMBB（增强移动宽带）、uRLLC（超高可靠超低延迟）、mMTC（海量机器类通信）。"
    },
    {
        "instruction": "如何在Python中实现定时任务？",
        "output": "Python定时任务实现：1.使用time.sleep()循环；2.sched模块；3.apscheduler库（推荐）：from apscheduler.schedulers.blocking import BlockingScheduler; scheduler.add_job(func, 'interval', seconds=10); scheduler.start()。"
    },
    {
        "instruction": "解释什么是区块链智能合约",
        "output": "智能合约是运行在区块链上的自动化脚本，满足条件时自动执行。以太坊是最流行的智能合约平台，使用Solidity语言开发。应用包括DeFi、NFT和供应链管理。"
    },
    {
        "instruction": "Python中的正则表达式有什么用途？",
        "output": "正则表达式用于模式匹配和文本处理，支持复杂的字符串操作。Python通过re模块实现：import re; result = re.search(pattern, string); re.findall(pattern, string)。用途：验证输入、提取信息、替换文本。"
    },
    {
        "instruction": "什么是虚拟现实（VR）和增强现实（AR）？",
        "output": "VR创建完全虚拟的环境，用户沉浸其中；AR将虚拟内容叠加到现实世界。VR需要头显设备，AR可通过手机或专用眼镜实现。应用包括游戏、培训和医疗。"
    },
    {
        "instruction": "如何在Python中处理大型CSV文件？",
        "output": "处理大型CSV使用分块读取：import pandas as pd; chunk_iter = pd.read_csv('large.csv', chunksize=1000); for chunk in chunk_iter: process(chunk)。或使用csv模块逐行读取，减少内存占用。"
    },
    {
        "instruction": "解释什么是设计模式",
        "output": "设计模式是解决软件设计中常见问题的最佳实践。创建型模式（单例、工厂）、结构型模式（适配器、装饰器）、行为型模式（观察者、策略）。Python常用模式：单例、工厂、装饰器、迭代器。"
    },
    {
        "instruction": "Python中的元类有什么作用？",
        "output": "元类是创建类的类，控制类的创建过程。用途：1.强制类属性；2.自动注册子类；3.修改类定义；4.实现ORM映射。通过type或继承type创建自定义元类。"
    },
    {
        "instruction": "什么是大数据处理框架？",
        "output": "大数据处理框架包括：1.Hadoop MapReduce：分布式批处理；2.Spark：内存计算框架；3.Flink：流处理框架；4.Hive：数据仓库工具；5.Kafka：分布式消息系统。用于处理大规模数据集。"
    },
    {
        "instruction": "如何在Python中实现线程安全？",
        "output": "Python线程安全实现：1.使用锁（threading.Lock()）；2.使用线程安全的数据结构（queue.Queue）；3.避免共享状态；4.使用threading.local()存储线程私有数据；5.使用原子操作。"
    },
    {
        "instruction": "解释什么是机器学习中的集成方法",
        "output": "集成方法组合多个模型提高性能：1.Bagging（如随机森林）：并行训练多个模型；2.Boosting（如AdaBoost、XGBoost）：顺序训练，关注错误样本；3.Stacking：组合不同类型模型的预测。"
    },
    {
        "instruction": "Python中的类型提示有什么作用？",
        "output": "类型提示（PEP 484）指定变量和函数参数/返回值类型：def add(a: int, b: int) -> int: return a + b。作用：提高代码可读性、支持静态类型检查（mypy）、IDE自动补全和重构。"
    },
    {
        "instruction": "什么是WebAssembly？",
        "output": "WebAssembly是低级二进制指令格式，允许高性能代码在浏览器中运行。支持C/C++/Rust等语言编译为wasm，在Web平台实现接近原生的性能，用于游戏、视频编辑等重型应用。"
    },
    {
        "instruction": "如何在Python中实现数据库迁移？",
        "output": "Python数据库迁移工具：1.Alembic（SQLAlchemy配套工具）；2.Django ORM迁移；3.Flask-Migrate。使用命令行创建和应用迁移脚本，管理数据库模式变更。"
    },
    {
        "instruction": "解释什么是REST和GraphQL",
        "output": "REST是资源为中心的API设计风格，使用HTTP方法；GraphQL是查询语言，允许客户端指定所需数据结构。GraphQL减少网络请求，REST实现简单。各有适用场景。"
    },
    {
        "instruction": "Python中的协程和线程有什么区别？",
        "output": "协程是用户态轻量级'线程'，由程序控制调度；线程是内核态，由操作系统调度。协程切换成本低，适合高并发I/O任务；线程适合CPU密集型任务，受GIL限制。"
    },
    {
        "instruction": "什么是数字孪生？",
        "output": "数字孪生是物理对象或系统的虚拟副本，通过实时数据同步模拟物理实体状态和行为。应用于制造业、医疗、城市规划等，实现预测性维护和优化决策。"
    },
    {
        "instruction": "如何在Python中进行性能分析？",
        "output": "Python性能分析工具：1.cProfile：函数级性能分析；2.line_profiler：行级性能分析；3.memory_profiler：内存使用分析；4.timeit：测量小段代码执行时间。使用这些工具定位性能瓶颈。"
    },
    {
        "instruction": "解释什么是机器学习中的迁移学习",
        "output": "迁移学习利用预训练模型解决新任务，将知识从一个领域迁移到另一个领域。优点：减少数据需求、加速训练、提高性能。常见应用：计算机视觉（使用ImageNet预训练模型）和NLP（使用BERT等）。"
    },
    {
        "instruction": "Python中的上下文变量有什么作用？",
        "output": "上下文变量（contextvars模块）在异步任务中共享状态，比线程局部变量更灵活。适用于异步框架中的请求上下文、事务ID跟踪等场景。使用：var = contextvars.ContextVar('name'); var.set(value); var.get()。"
    },
    {
        "instruction": "什么是云计算的部署模型？",
        "output": "云计算部署模型：1.公有云（AWS、Azure）：第三方提供，多租户；2.私有云（OpenStack）：企业自建，单租户；3.混合云：公有云+私有云；4.社区云：特定组织共享。选择基于安全、成本和控制需求。"
    },
    {
        "instruction": "如何在Python中实现自定义异常？",
        "output": "Python自定义异常继承Exception类：class MyException(Exception): pass。可添加自定义属性和方法：class ValidationError(Exception): def __init__(self, field, message): self.field = field; self.message = message。使用：raise ValidationError('email', '无效邮箱格式')。"
    },
    {
        "instruction": "解释什么是数据湖和数据仓库",
        "output": "数据湖存储原始、未处理的所有数据（结构化和非结构化）；数据仓库存储结构化、处理过的数据，针对分析优化。数据湖适合探索性分析，数据仓库适合报表和决策支持。"
    },
    {
        "instruction": "Python中的描述符有什么作用？",
        "output": "描述符是实现__get__、__set__或__delete__方法的对象，用于自定义属性访问。用途：1.属性验证；2.延迟计算；3.数据绑定；4.ORM字段映射。Python内置描述符：property、classmethod、staticmethod。"
    },
    {
        "instruction": "什么是自然语言生成（NLG）？",
        "output": "NLG是从数据生成自然语言文本的AI技术，将结构化数据转换为人类可读的语言。应用包括自动报告生成、聊天机器人回复、机器翻译和摘要生成。基于规则或机器学习方法实现。"
    },
    {
        "instruction": "如何在Python中使用正则表达式验证邮箱格式？",
        "output": "邮箱验证正则表达式：import re; pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'; def is_valid_email(email): return re.match(pattern, email) is not None。注意：完整邮箱验证复杂，生产环境建议使用专门库如email-validator。"
    },
    {
        "instruction": "解释什么是软件开发生命周期（SDLC）",
        "output": "SDLC是软件开发的系统化流程，包括：1.规划；2.需求分析；3.设计；4.开发；5.测试；6.部署；7.维护。模型包括瀑布模型、敏捷、迭代模型等，确保软件质量和按时交付。"
    },
    {
        "instruction": "Python中的functools模块有哪些常用功能？",
        "output": "functools模块常用功能：1.wraps：保留函数元数据的装饰器；2.lru_cache：函数结果缓存；3.partial：部分应用函数；4.reduce：累积计算；5.total_ordering：自动实现比较方法。"
    },
    {
        "instruction": "什么是增强学习中的Q-learning？",
        "output": "Q-learning是无模型强化学习算法，学习动作价值函数Q(s,a)（状态s下执行动作a的预期奖励）。通过贝尔曼方程更新Q值，找到最优策略。适用于离散状态和动作空间的问题。"
    },
    {
        "instruction": "如何在Python中实现一个简单的Web服务器？",
        "output": "Python简单Web服务器：1.使用http.server模块：python -m http.server 8000；2.使用Flask：from flask import Flask; app = Flask(__name__); @app.route('/') def home(): return 'Hello'; app.run()。Flask更适合开发实际应用。"
    },
    {
        "instruction": "解释什么是机器学习中的特征缩放",
        "output": "特征缩放标准化或归一化特征值，使不同量级特征具有可比性。常用方法：1.标准化（StandardScaler）：均值为0，标准差为1；2.归一化（MinMaxScaler）：缩放到[0,1]范围。对距离-based算法（SVM、KNN）和梯度下降优化至关重要。"
    },
    {
        "instruction": "Python中的collections模块有哪些有用的数据结构？",
        "output": "collections模块数据结构：1.defaultdict：带默认值的字典；2.Counter：计数容器；3.deque：双端队列；4.namedtuple：命名元组；5.OrderedDict：有序字典（Python 3.7+普通字典已有序）；6.ChainMap：合并多个字典。"
    },
    {
        "instruction": "什么是元宇宙？",
        "output": "元宇宙是持久的虚拟共享空间，融合物理世界和数字世界。基于虚拟现实、增强现实、区块链等技术，支持社交、工作、娱乐等活动。特点：持续性、实时性、共享性和沉浸感。"
    }
]

# 3. 将指令转换为模型输入格式
def format_instruction(examples):
    texts = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        # 使用简单的指令模板
        text = f"### 指令:\n{instruction}\n\n### 回答:\n{output}"
        texts.append(text)
    # 分词
    tokenized = tokenizer(texts, padding="max_length", truncation=True, max_length=128)
    # 标签与输入相同（自回归训练）
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# 创建数据集
dataset = Dataset.from_dict({
    "instruction": [ex["instruction"] for ex in instructions],
    "output": [ex["output"] for ex in instructions]
})

# 格式化数据集
tokenized_dataset = dataset.map(format_instruction, batched=True)

# 4. 配置训练参数（优化版）
training_args = TrainingArguments(
    output_dir="./instruction_tuned_model",
    overwrite_output_dir=True,
    num_train_epochs=50,  # 大幅增加训练轮次
    per_device_train_batch_size=4,  # 适当增大批次大小
    save_steps=100,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=10,
    learning_rate=5e-5,  # 提高学习率
    weight_decay=0.01,  # 添加权重衰减防止过拟合
    warmup_steps=100,  # 热身步数
    ddp_find_unused_parameters=False
)

# 5. 创建数据收集器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False  # 非掩码语言模型
)

# 6. 创建Trainer并训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("开始指令微调...")
trainer.train()

# 7. 保存微调后的模型
# 替换trainer.save_model为直接保存模型和分词器
model.cpu()  # 将模型转移到CPU以避免分布式张量问题
model.save_pretrained("./instruction_tuned_model")
tokenizer.save_pretrained("./instruction_tuned_model")
print("模型微调完成并保存!")

# 8. 测试微调后的模型
def generate_answer(model, tokenizer, instruction, max_length=100):
    # 格式化输入指令
    input_text = f"### 指令:\n{instruction}\n\n### 回答:\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # 生成回答（优化参数）
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=len(inputs.input_ids[0]) + max_length,
            temperature=0.7,  # 适度随机性
            top_p=0.9,        #  nucleus sampling
            repetition_penalty=1.5,  # 增强重复惩罚
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            num_return_sequences=1
        )
    
    # 解码并提取回答部分
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_start = output_text.find("### 回答:\n") + len("### 回答:\n")
    return output_text[answer_start:]

# 测试模型
test_instructions = [
    "什么是光合作用？",
    "Python中的列表推导式如何使用？",
    "太阳系有哪些行星？"
]

print("\n测试微调后的模型:")
for instruction in test_instructions:
    answer = generate_answer(model, tokenizer, instruction)
    print(f"指令: {instruction}")
    print(f"回答: {answer}\n")