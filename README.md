# DCSIL-MDA
DCSIL-MDA: Debiased Causal Semantic Interaction Learning for Microbe-Drug Association Prediction


## Overview
DCSIL-MDA is a multi-view representation learning framework designed for microbe–drug association prediction. It integrates causal semantic modeling, collaborative interaction learning, and heterogeneous structural representation within a unified architecture, enabling more accurate identification of potential associations.


## Result model
The prediction results of the model on the MDAD dataset are available in the `result_MDAD` directory.


## Data
experiments were conducted on three publicly available datasets, including MDAD[1], aBiofilm[2], and DrugVirus[3]. 

[1]	Sun Y-Z, Zhang D-H, Cai S-B et al. MDAD: a special resource for microbe-drug associations, Frontiers in cellular and infection microbiology 2018;8:424.

[2]	Rajput A, Thakur A, Sharma S et al. aBiofilm: a resource of anti-biofilm agents and their potential implications in targeting antibiotic drug resistance, Nucleic acids research 2018;46:D894-D900.

[3]	Andersen PI, Ianevski A, Lysvand H et al. Discovery and development of safe-in-man broad-spectrum antiviral agents, International Journal of Infectious Diseases 2020;93:268-276.


## Training and Evaluation
To run the model, execute: python main.py


