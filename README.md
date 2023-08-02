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
The Tensor-layout in all classes is assumed to be Time x Samples x Processes.

The fish farm can be customized completely by subclassing the following classes and overwriting their main class functions:
- [Harvesting Costs](Python/Harvest.py)
    - Abstract class: Harvest
    - Abstract methods:
        - sample: takes batch_size as input
        - setgen: sets random number generator for numpy or tensorflow
        - cost: takes output of sample as input
        - totalCost: takes output of sample as input, as well as current biomass
    - Currently implemented:
        - Harvest: implements constant harvesting costs
- [Salmon Growth](Python/Growth.py)
    - Abstract class: Growth
    - Abstract methods:
        - sample: takes batch_size as input
        - setgen: sets random number generator for numpy or tensorflow
        - weight: takes output of sample as input
    - Currently implemented:
        - Bertalanffy: Bertalanffy's deterministic growth model
- [Salmon Price](Python/Price.py)
    - Abstract class: Price
    - Abstract methods:
        - sample: takes batch_size as input
        - setgen: sets random number generator for numpy or tensorflow
        - price: takes output of sample as input
    - Currently implemented:
        - Price: implements price model by using [commodity](Python/Commodity.py) spot price model, assumes that spot price has the following dimensions: time x simulation x factors, where the first factor dimension is the spot price and e.g. the second is the convenience yield
- [Feeding Costs](Python/Feed.py)
    - Abstract class: Feed
    - Abstract methods:
        - sample: takes batch_size as input
        - setgen: sets random number generator for numpy or tensorflow
        - cost: takes output of sample as input
        - cumtotalCost: takes output of sample as input, as well as current weight and current number of fish
    - Currently implemented:
        - Feed: implements constant feeding costs
        - StochFeed: implements feeding costs depending on [commodity](Python/Commodity.py) spot price model, assumes that spot price has the following dimensions: time x simulation x factors, where the first factor dimension is the spot price and e.g. the second is the convenience yield
        - DetermFeed: is the expectation of StochFeed
- [Mortality](Python/Mortality.py)
    - Abstract class: Mortality
    - Abstract methods:
        - sample: takes batch_size as input
        - setgen: sets random number generator for numpy or tensorflow
        - populationSize: takes output of sample as input
        - treatmentCosts: takes output of sample as input
    - Currently implemented:
        - ConstMortality: constant deterministic mortality model: $H_0 \exp(-m t)$
        - HostParasite: stochastic host-parasite model
        - DetermHostParasite: expectation of HostParasite
        - Poisson: stochastic Poisson model
        - DetermPoisson: expectation of Poisson

You can use any combination of the currently implemented models by selecting it in [main](Python/main.m).


# Data
The data provided in this repository may not be used for anything else, than testing the algorithms in this repository. The license for the code does not extend to the datasets.

The most recent data concerning salmon lice can be downloaded from [barentswatch.no](https://www.barentswatch.no/nedlasting/fishhealth) while agreeing to their terms of usage.