# CoTKG-IDS: 基于思维链知识图谱的入侵检测系统

## 概述
CoTKG-IDS 是一个先进的网络入侵检测系统，它结合了思维链(Chain of Thought, CoT)推理、知识图谱和GraphSAGE技术，以提供增强的检测能力和可解释性。该系统通过利用图深度学习和知识表示技术，能够有效地检测和分类各种类型的网络入侵行为。

## 主要特点
- 🧠 基于思维链(CoT)的可解释推理检测
- 🕸️ 基于网络流量数据的动态知识图谱构建
- 📊 基于GraphSAGE的网络分析与模式检测
- 🔍 具有自动选择功能的高级特征工程
- ⚖️ 智能数据平衡处理不平衡攻击类别
- 🎯 高精度多类攻击检测
- 📈 全面的可视化分析工具
- 🔄 实时处理能力
- 🛠️ 模块化架构便于扩展

## 系统架构
```
输入数据 → 特征工程 → 知识图谱构建 → GraphSAGE模型 → 攻击检测
   ↓          ↓           ↓             ↓           ↓
 预处理 → 特征选择 → 图嵌入生成 → 思维链推理 → 可解释性分析
```

## 安装说明

### 环境要求
- Python 3.7+
- Neo4j数据库 4.4+
- PyTorch 1.9+
- CUDA（可选，用于GPU加速）
- 建议8GB以上内存
- 建议50GB以上磁盘空间（用于完整数据集）

### 环境配置
1. 克隆代码仓库：
```bash
git clone https://github.com/chenxingqiang/cotkg-ids.git
cd cotkg-ids
```

2. 创建并激活虚拟环境：
```bash
# 使用venv
python -m venv venv
source venv/bin/activate  # Linux/Mac系统
venv\Scripts\activate     # Windows系统

# 或使用conda
conda create -n cotkg-ids python=3.9
conda activate cotkg-ids
```

3. 安装依赖：
```bash
pip install -r requirements.txt

# 如果pip下载速度较慢，可以使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Neo4j数据库配置
1. 安装Neo4j：
```bash
# 推荐使用Docker安装
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    -d neo4j:4.4

# 或从neo4j.com下载安装包安装
```

2. 配置Neo4j：
- 访问 http://localhost:7474
- 使用默认凭据登录（用户名/密码：neo4j/neo4j）
- 根据提示修改密码
- 在config/config.py中更新您的凭据

### 数据准备
1. 下载数据集：
```bash
python download_data.py
```

2. 验证数据完整性：
```bash
python test_pipeline.py --mode data_check
```

## 使用说明

### 快速开始
```bash
# 使用默认设置运行完整流程
python run.py

# 仅运行训练
python run.py --mode train

# 仅运行测试
python run.py --mode test

# 使用测试配置运行
python run.py --test
```

### Python API使用
```python
from src.main import run_full_pipeline
from config.config import DEFAULT_CONFIG

# 使用默认配置
results = run_full_pipeline()

# 自定义配置
config = DEFAULT_CONFIG.copy()
config['model']['graphsage'].update({
    'hidden_channels': 64,
    'num_layers': 3,
    'dropout': 0.2
})
results = run_full_pipeline(config=config)
```

### 配置说明
系统可以通过 `config/config.py` 进行配置。主要配置项包括：

```python
DEFAULT_CONFIG = {
    'model': {
        'graphsage': {
            'hidden_channels': 32,    # 隐藏层通道数
            'num_layers': 2,          # 层数
            'dropout': 0.3,           # dropout率
            'learning_rate': 0.01,    # 学习率
            'weight_decay': 0.0005    # 权重衰减
        }
    },
    'training': {
        'epochs': 20,                 # 训练轮数
        'batch_size': 16,             # 批次大小
        'early_stopping': {
            'patience': 5,            # 早停耐心值
            'min_delta': 0.01         # 最小变化阈值
        },
        'validation_split': 0.2       # 验证集比例
    },
    'data': {
        'balancing': {
            'method': 'smote',        # 数据平衡方法
            'random_state': 42        # 随机种子
        }
    },
    'neo4j': {
        'uri': 'bolt://localhost:7687',
        'username': 'neo4j',
        'password': 'password'
    }
}
```

## 项目结构
```
cotkg-ids/
├── config/                 # 配置文件
├── data/                  # 数据存储
│   ├── raw/              # 原始数据集
│   └── processed/        # 处理后的数据
├── logs/                 # 日志文件
├── notebooks/            # Jupyter笔记本
├── results/              # 输出文件
│   ├── models/          # 保存的模型
│   └── visualizations/  # 生成的可视化
├── src/                 # 源代码
│   ├── data_processing/ # 数据处理模块
│   ├── knowledge_graph/ # 知识图谱相关代码
│   ├── models/         # 机器学习模型
│   └── visualization/  # 可视化工具
└── tests/              # 测试文件
```

## 性能指标
系统在多个指标上进行评估：
- 检测准确率：测试集上约98%
- 误报率：低于1%
- 处理速度：约1000流/秒
- 内存使用：标准数据集约4GB

## 常见问题解决

### 常见问题
1. Neo4j连接问题：
```bash
# 检查Neo4j状态
docker ps | grep neo4j
# 或
service neo4j status
```

2. CUDA问题：
```python
import torch
print(torch.cuda.is_available())  # 如果CUDA正确配置，应返回True
```

3. 内存问题：
- 在配置中减小batch_size
- 对大数据集使用数据采样
- 必要时启用交换空间

### 错误信息说明
- "Neo4j connection failed"：检查Neo4j凭据和服务状态
- "CUDA out of memory"：减小批次大小或模型大小
- "File not found"：确保数据集已下载并位于正确位置

## 参与贡献
1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m '添加某个特性'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

### 开发指南
- 遵循PEP 8编码规范
- 为新特性添加测试
- 更新文档
- 使用类型提示
- 添加适当的错误处理

## 开源协议
MIT License - 查看 [LICENSE](LICENSE)

## 引用
```bibtex
@software{cotkg_ids2024,
  author = {Chen, Xingqiang},
  title = {CoTKG-IDS: Chain of Thought Knowledge Graph IDS},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/chenxingqiang/cotkg-ids}
}
```

## 联系方式
陈星强  
邮箱: chen.xingqiang@iechor.com  
GitHub: [@chenxingqiang](https://github.com/chenxingqiang)

## 特别说明
1. 如果您在中国大陆使用本系统，某些依赖包的下载可能较慢，建议使用国内镜像源。
2. 建议在使用前确保您的网络环境能够稳定访问GitHub和相关资源。
3. 如有任何问题，欢迎通过Issue或邮件联系。 