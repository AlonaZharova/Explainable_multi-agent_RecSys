# Explainable Multi-Agent Recommendation System for Energy-Efficient Smart Home

A digital companion to the research paper 

```
Alona Zharova, Annika Boer, Julia Knoblauch, Kai Ingo Schewina, Jana Vihs (2022).
Explainable Multi-Agent Recommendation System for Energy-Efficient Smart Home.
[...]
```
The paper is available at: [...]

![results](/structure.jpg)

## Summary 

Energy use has steadily increased worldwide over the last decades and will likely increase further. A possible solution to tackle the energy efficiency problem is a behavioral change to energy consumption. Recommender systems can suggest actions to improve energy efficiency that facilitates such behavioral change. Generating recommendations as explainable can help achieve a higher persuasiveness and, therefore, the higher effectiveness of the recommender systems. The existing research on explainability in recommender systems is very scarce. Moreover, most existing approaches are not applicable to the smart home area because of the missing data structures. 

We extend the approach by Riabchuk et al. (2022) and suggest an explainable multi-agent recommendation system for energy efficiency in private households. We extend the multi-agent system to include seven agents: electricity price agent, data preparation agent, user availability agent, device usage agent, device load agent, explainability agent, and recommendation agent. We improve the performance issues from Riabchuk et al. (2022) by testing multiple models for the prediction tasks such as KNN, XGBoost, AdaBoost, Random Forest, Logistic Regression, and Explainable Boosting Machine. We focus on the explainability of the recommendations by applying three approaches, i.e., local interpretable model-agnostic explanations (LIME), Shapley additive explanations (SHAP), and the explainable boosting machine (EBM). Our results show a substantial increase in performance while at the same time opening up the "black box" of the recommendations. We provide users with comprehensive, understandable, and persuasive explanations to achieve behavioral change with regard to energy efficiency. 

Riabchuk, V., Hagel, L., Germaine, F. and Zharova, A. 2022. Utility-Based Context-Aware Multi-Agent Recommendation System for Energy Efficiency in Residential Buildings. arXiv preprint. DOI: doi.org/10.48550/arXiv.2205.02704.

We provide a comprehensive tutorial in Jupyter Notebook with code in Python for all the steps described in this paper and beyond.

**Keywords:** recommendation system, multi-agent system, explainable AI, load shifting, energy consumption behavior.

## Data

We use the REFIT Electrical Load Measurements data ([Murray et al., 2017](https://www.nature.com/articles/sdata2016122)) to analyze our recommender system. The data contains the energy consumption of nine different devices used in 20 households in the United Kingdom from 2013 to 2015. 

For the day-ahead prices provided by the Price Agent, we access the online database for industry day-ahead prices for the United Kingdom ([ENTSO-E, 2015](https://transparency.entsoe.eu/transmission-domain/r2/dayAheadPrices/show)). 



