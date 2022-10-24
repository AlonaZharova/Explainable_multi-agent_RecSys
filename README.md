# Explainable Multi-Agent Recommendation System for Energy-Efficient Smart Home

A digital companion to the preprint

```
Alona Zharova, Annika Boer, Julia Knoblauch, Kai Ingo Schewina, Jana Vihs (2022).
Explainable Multi-Agent Recommendation System for Energy-Efficient Smart Home.
arXiv preprint. DOI: doi.org/10.48550/arXiv.2210.11218.
```
The paper is available at: [arXiv](https://doi.org/10.48550/arXiv.2210.11218)

The paper was accepted to the [NeurIPS 2022 Tackling Climate Change with Machine Learning Workshop](https://nips.cc/virtual/2022/workshop/49964).

![results](/recommendation_user.jpg)

## Summary 

Energy use has steadily increased worldwide over the last decades and will likely increase further. A possible solution to tackle the energy efficiency problem is a behavioral change to energy consumption. Recommender systems can suggest actions to improve energy efficiency that facilitates such behavioral change. Generating recommendations as explainable can help achieve a higher persuasiveness and, therefore, the higher effectiveness of the recommender systems. The existing research on explainability in recommender systems is very scarce. Moreover, most existing approaches are not applicable to the smart home area because of the missing data structures. 

We extend the approach by Riabchuk et al. (2022) and suggest an explainable multi-agent recommendation system for energy efficiency in private households. We extend the multi-agent system to include seven agents: electricity price agent, data preparation agent, user availability agent, device usage agent, device load agent, explainability agent, and recommendation agent. We improve the performance issues from Riabchuk et al. (2022) by testing multiple models for the prediction tasks such as KNN, XGBoost, AdaBoost, Random Forest, Logistic Regression, and Explainable Boosting Machine. We focus on the explainability of the recommendations by applying three approaches, i.e., local interpretable model-agnostic explanations (LIME), Shapley additive explanations (SHAP), and the explainable boosting machine (EBM). Our results show a substantial increase in performance while at the same time opening up the "black box" of the recommendations. We provide users with comprehensive, understandable, and persuasive explanations to achieve behavioral change with regard to energy efficiency. 

Riabchuk, V., Hagel, L., Germaine, F. and Zharova, A. 2022. Utility-Based Context-Aware Multi-Agent Recommendation System for Energy Efficiency in Residential Buildings. arXiv preprint. DOI: doi.org/10.48550/arXiv.2205.02704.

We provide a comprehensive tutorial in Jupyter Notebook with code in Python for all the steps described in this paper and beyond.

**Keywords:** recommendation system, multi-agent system, explainable AI, load shifting, energy consumption behavior.

## Data

We use the REFIT Electrical Load Measurements data ([Murray et al., 2017](https://www.nature.com/articles/sdata2016122)) to analyze our recommender system. The data contains the energy consumption of nine different devices used in 20 households in the United Kingdom from 2013 to 2015. 

For the day-ahead prices provided by the Price Agent, we access the online database for industry day-ahead prices for the United Kingdom ([ENTSO-E, 2015](https://transparency.entsoe.eu/transmission-domain/r2/dayAheadPrices/show)). 

## Project structure
````
    ├── README.md                                             # this readme file
    │    
    ├── requirements.txt                                      # this file may be used to create an environment
    │
    ├── structure.jpg                                         # figure of the multi-agent structure
    │
    ├── code.                                                 # agent notebooks + .py scripts
    │   ├── Project.ipynb
    │   ├── agents.py
    │   ├── grid_search.py
    │   └── helper_functions.py
    │
    ├── data                                                  # data
    │   ├── processed_pickle                                           
    │   |   ├── activity_df.pkl                                             
    │   |   ├── df_th.pkl                                           
    │   |   ├── load_df.pkl                                           
    │   |   ├── price_df.pkl                                           
    │   |   ├── usage_df.pkl                                                   
    │   ├── Day-ahead Prices_201501010000-201601010000.csv                                                     
    │   ├── REFIT_Readme.txt
    │
    └── export                                                # path for exporting configurations and intermediate results
        ├── 1_config.json                                     # configurations used for evaluating households 1 to 10
        ├── [...]
        └── 10_config.json
````

### Adding Data:
 - Due to file size restrictions, we did not include any of the REFIT: Electrical Load Measurements data (Murray et al. 2017)
 - These files can be accessed using the following link: https://www.doi.org/10.15129/9ab14b0e-19ac-4279-938f-27f643078cec
 - After downloading the clean household data needs to be copied to ./data

## Citation

If you use this code in your research, please cite our [paper](https://doi.org/10.48550/arXiv.2210.11218).

```
@misc{ZBKSV2022,
  title = {Explainable Multi-Agent Recommendation System for Energy-Efficient Smart Home},
  author = {Zharova, Alona and Boer, Annika and Knoblauch, Julia and Schewina, Kai Ingo and Vihs, Jana},
  publisher = {arXiv},
  year = {2022},
  doi = {10.48550/arXiv.2210.11218}  
}
```

## Contact
- Alona Zharova, alona.zharova@hu-berlin.de
