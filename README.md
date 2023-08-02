# Aquaculture with Stochastic Mortality
This repository is complementing [TODO](https://www.arxiv.org/) for investigating the effect of stochastic mortality to optimal harvesting rules in aquaculture farms. In particular, we chose salmon farms for our investigation.

## Installation
There is no installation required, please download the code and run the main files. For Python you need all the necessary dependencies to run Tensorflow 2.x. There is also a Jupyter notebook available for testing the Python code in Colab.

[![Open In Colab TODO](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kevinkamm/AquacultureStochasticFeeding/blob/main/Python/main.ipynb)

## Code and Usage
The code is structured as follows: 

For the data preprocessing and calibration of the mortality models to the salmon lice data we used Matlab with its (Global) Optimization Toolbox. 

### Python
Optimal Stopping:
1. Run the jupyter notebook [main](Python/main.m) with your custom model parameters to find the optimal harvesting time. Also returns a comparison of stopping rules with deterministic and stochastic feeding costs using a pathwise comparison.

### Matlab
Data Preprocessing:
1. Run [preprocessingRaw](Matlab/Data/preprocessingRaw.m) to preprocess salmon lice data.
2. Run [preprocessWeeklyWithoutAlign](Matlab/Data/preprocessWeeklyWithoutAlign.m) to process the preprocessed salmon lice data.

Calibration:
- [Host-Parasite Model](Matlab/HostParasite/): 
    1. Run [mainHostParasite](Matlab/HostParasite/mainHostParasite.m) to calibrate the host-parasite model.
- [Poisson Model](Matlab/Poisson/): 
    1. Run [mainPoisson](Matlab/Poisson/mainHostParasite.m) to calibrate the Poisson model.


## Fish Farm Objects
The fish farm can be customized completely by subclassing the following classes and overwriting their main class functions:
- [Harvesting Costs](Python/Harvest.py)
- [Feeding Costs](Python/Feed.py)
- [Salmon Growth](Python/Growth.py)
- [Salmon Price](Python/Price.py)
- [Mortality](Python/Mortality.py)


# Data
The data provided in this repository may not be used for anything else, than testing the algorithms in this repository. The license for the code does not extend to the datasets.

The most recent data concerning salmon lice can be downloaded from [barentswatch.no](https://www.barentswatch.no/nedlasting/fishhealth) while agreeing to their terms of usage.