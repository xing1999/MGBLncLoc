# MGBLncLoc
An Ensemble Deep Learning Framework for Multi-label LncRNA Subcellular Localization with Innovative Encoding Strategy

# Requirements
```bash
absl-py==0.9.0
astor==0.8.1
bio==0.1.0
certifi==2020.4.5.1
cycler==0.10.0
gast==0.3.3
grpcio==1.28.1
h5py==2.10.0
joblib==0.14.1
Keras==2.2.4
Keras-Applications==1.0.8
Keras-Preprocessing==1.1.0
kiwisolver==1.2.0
Markdown==3.2.1
numpy==1.18.3
pandas==1.0.3
protobuf==3.11.3
pyparsing==2.4.7
python-dateutil==2.8.1
pytz==2020.1
PyYAML==5.3.1
scikit-learn==0.22.2.post1
scipy==1.2.1
seaborn==0.11.0
six==1.14.0
sklearn==0.0
tensorboard==1.12.2
tensorflow-gpu==1.12.0
```
# Detailed installation and setup guide

Make sure the following is included in the user: (1) All necessary dependencies and their installation commands. (2) Specific steps for installing a virtual environment and installing dependencies.

## Installation guide
### Cloning the repository
```bash
git clone https://github.com/xing1999/MGBLncLoc.git
cd MGBLncLoc
```
### 1. Create a virtual environment and install dependencies
```bash
python3 -m venv venv
source venv/bin/activate  # Using `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```
### 2. Configure the pre-trained model code environment
### 3. Preparation of input data

The input data should be a TSV file containing lncRNA sequences. Below is an example file `input_data.tsv`:

index/ 	  label/ 	  text
```bash
1	0	TTCTGGACACTATTCTATTTATCTATGTTTTTATTCTTTCACCAATTCCACATTGTCTTGATTACTATAACTTTATAGTAAGTCTTGAAATTAAGTAATGTGAGTCCTCTGACTTTGTTGTTCTTCTATATTGTATTGCCTATGCTCAGCTACTCTTATCCATATAAAATATACAGTAGTTTGTTGATATCTAGAAAATACCTTGATGAGAATTTGATTGG
2	1	AGACTAGGGATGTCCTGAGAACTATATTGATAGCTGCGACCTGTGTCTCGGTTGCTTCTCTCTTGTTCAGTGCTGCTTCCTTTACTCTCATAGGTGTTTGCTCCCTAAAATATTCCCCAGTCTGCTTCCTGGGGAAACCCAACCTAAAACTCCTAACTGACCTAAAACAGGCTGCTGGGGTCTAGCACTCTTCAGCTAGGACTTCACTTCTTCCCTCATGG
3	2	GTGACTTCAACTGAATAAATTTGAATTTCTGTAGGGAGTAAAGAATCAAAACACCTATTTAAAGACTGCAAAATATGATAATTATTTTTAAAGTAATTGATTAAACCTGGTAGGTTTTCCCAAAATGAAAAACAATCAGTTCTAAAACCAAAGCTGATTTTTAGAAAATGTGAAAATGTAAATCAACCCTATCCATAATAGATTCTCTAAAACTTTATCTT
4	3	TGTCACCATGCCCTGTTAATTTTTGTTTGTTTGTTTTGTTTTTTGTTTTGTTTTGTTTCGTTTTTTGTTTTTTTTTTTTGAGACGGAGTCTTGCTCTGTCGCCCAGGCTGGAGTGCAGTGGCCCCGATCTCGGCTCACCGCAAGCTCCGCCTCGCGGGTTCATGCCATTCTCCGGCCTCAGCCTCCCGAGTAGCTGGGACTACAGGCATCCGCCACCACAC
```
Place your input data file in the `data` folder.
### 4. Run the prediction
Use the following command to run the prediction script:
```bash
python train.py
```
### 5. Code structure and function descriptions
#### Code structure
- `encode.py`: LncRNA sequence data encoder script.
- `model.py`: training model construction code.
- `mian.py`: overall model code.
- `trian.py`: code section for model training sequence data.
- `test.py`: code section for model test sequence data.
- `utils.py`: module containing auxiliary functions and tools.
### 6. Frequently Asked Questions (FAQ)
Q1: How to solve the dependency installation problem?

A: Please make sure you are using a virtual environment and run the following command:
```bash
pip install -r requirements.txt
```
Q2: What should I do if I get an error when running the prediction script?

A: Please check if the input file path and model file path are correct. If the problem still exists, please contact the author for detailed error information and we will try our best to answer it.
