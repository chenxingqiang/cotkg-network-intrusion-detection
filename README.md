
# CoT-KG Network Intrusion Detection using Knowledge Graph and GraphSAGE

This project implements a network intrusion detection system using Chain of Thought (CoT), knowledge graphs and GraphSAGE model on the CICIDS2017 dataset. The Chain of Thought approach is used to enhance the knowledge graph construction and improve the interpretability of the detection process.

## Key Features

- Chain of Thought (CoT) enhanced knowledge graph construction
- GraphSAGE-based network intrusion detection
- Interpretable AI techniques for explaining detection results
- Integration of domain knowledge with machine learning

## Data Download and Preparation

The CICIDS2017 dataset is used in this project. Follow these steps to download and prepare the data:

 1. Clone this repository:

```bash
git clone https://github.com/chenxingqiang/cotkg-network-intrusion-detection.git
cd cotkg-network-intrusion-detection
```

 2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

 3. Run the data download script:

```bash
python src/download_data.py
```

This script will:

- Download the MachineLearningCSV.zip file from the CICIDS2017 dataset

- Check the integrity of the downloaded file

- Extract the contents to the `data/raw/` directory

Note: The download might take some time as the file is about 224MB.

4. After running the script, the data will be available in the `data/raw/MachineLearningCVE/` directory.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

After preparing the data, you can run the main script to train and evaluate the model:

```bash
python src/main.py

```

This script will:

- Load and preprocess the data
- Perform feature engineering
- Construct the knowledge graph
- Train the GraphSAGE model
- Evaluate the model
- Generate explanations for the predictions

## Note

The raw data files are large and are not included in the git repository. They will be downloaded when you run the `download_data.py` script. If you need to share the project, others can use the same script to download the data.

## Author

Chen Xingqiang
Hanghzou Turing AI Co.,Ltd.
Email: <chen.xingqiang@iechor.com>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- CICIDS2017 dataset: <https://www.unb.ca/cic/datasets/ids-2017.html>
