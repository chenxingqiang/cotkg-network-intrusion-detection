# CoTKG-IDS: Chain of Thought Knowledge Graph Intrusion Detection System

## Overview
CoTKG-IDS is an advanced network intrusion detection system that combines Chain of Thought (CoT) reasoning with knowledge graphs and GraphSAGE for enhanced detection capabilities and interpretability.

## Key Features
- 🧠 Chain of Thought (CoT) enhanced reasoning
- 🕸️ Dynamic knowledge graph construction
- 📊 GraphSAGE-based network analysis
- 🔍 Advanced feature engineering
- ⚖️ Intelligent data balancing
- 🎯 Multi-class attack detection
- 📈 Comprehensive visualization

## Architecture
```
Input Data → Feature Engineering → Knowledge Graph Construction → GraphSAGE Model → Attack Detection
    ↓               ↓                       ↓                         ↓                ↓
Preprocessing → Feature Selection → Graph Embeddings → Chain of Thought → Interpretability
```

## Installation

### Prerequisites
- Python 3.7+
- Neo4j Database
- PyTorch
- CUDA (optional, for GPU support)

### Quick Start
```bash
# Clone repository
git clone https://github.com/chenxingqiang/cotkg-ids.git
cd cotkg-ids

# Setup environment
./env_manager.sh setup

# Setup Neo4j
./setup_neo4j.sh

# Run the system
python run.py
```

### Docker Setup
```bash
docker-compose up -d
```

## Usage

### Basic Usage
```python
from src.main import run_full_pipeline

# Run complete pipeline
results = run_full_pipeline()
```

### Custom Configuration
```python
from src.config.config import DEFAULT_CONFIG

# Modify configuration
config = DEFAULT_CONFIG.copy()
config['model_params']['hidden_channels'] = 128
config['training_params']['epochs'] = 200

# Run with custom config
results = run_full_pipeline(config)
```

## Components

### Knowledge Graph Construction
- Dynamic graph building
- Feature-based node linking
- Similarity-based edge creation
- Automatic node mapping

### GraphSAGE Model
- Multi-layer architecture
- Dropout regularization
- Self-loop handling
- Automatic data preparation

### Data Processing
- Advanced feature engineering
- Intelligent data balancing
- Automated preprocessing
- Robust error handling

## Results Directory Structure
```
results/
├── feature_importance.csv   # Feature importance rankings
├── model_stats.txt         # Model performance statistics
├── visualizations/         # Generated visualizations
└── models/                # Saved model checkpoints
```

## Development

### Running Tests
```bash
python test_pipeline.py
```

### Code Style
```bash
# Format code
black src/

# Run linter
flake8 src/
```

### Version Management
```bash
# Bump version
./bump_version.sh patch  # or minor/major
```

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
MIT License - see [LICENSE](LICENSE)

## Citation
```bibtex
@software{cotkg_ids2024,
  author = {Chen, Xingqiang},
  title = {CoTKG-IDS: Chain of Thought Knowledge Graph IDS},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/chenxingqiang/cotkg-ids}
}
```

## Contact
Chen Xingqiang  
Email: chen.xingqiang@iechor.com
