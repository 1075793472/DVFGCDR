#  DVFGCDR
DVFGCDR: a dual-view fusion graph neural network for cancer drug response prediction
Some big data are included in the link below.\
[https://pan.baidu.com/s/1AnssTTJISnKcGAWz8fXwxQ?pwd=99q4](https://pan.baidu.com/s/1S_J6xaG6C2C9ZazrYE4F-Q?pwd=5a8p)
# Install
To use DSHCNet you must make sure that your python version is greater than 3.7. If you don’t know the version of python you can check it by:
```python
python
>>> import platform
>>> platform.python_version()
'3.7.13'
```
# Train and test DVFGCDR
1.Download the data from the Baidu Cloud link. \
2.Unzip the downloaded data (.zip file) from Baidu Cloud, and copy all files from the unzipped folder to the same level directory as the main.py file in the repository.
# Parameters
The parameters of the methods we compare come from the settings of each author in the paper \
DeepTTA: https://github.com/jianglikun/DeepTTC \
DeepCDR: https://github.com/kimmo1019/DeepCDR \
MOLI:    https://github.com/hosseinshn/MOLI \
tCNNS:   https://github.com/Lowpassfilter/tCNNS-Project
# Environment Requirement
The required packages are as follows:
- scikit-learn            0.22.1
- scipy                   1.4.1
- joblib==1.1.0
- numpy==1.22.3
- optlang==1.5.2
- pandas==1.3.5
- torch                   1.12.1
- torch-cluster           1.6.0+pt112cu113
- torch-geometric         2.6.1
- torch-scatter           2.1.0+pt112cu113
- torch-sparse            0.6.15+pt112cu113
- torchaudio              0.12.1
- torchvision             0.13.1
- tqdm                    4.66.5

