# SAGA_tab - link prediction on semantic graphs
SAGA_tab is an approach for the creation of a semantic
description of the table starting from a partial or complete initial semantic annotation. 
It requires a domain Ontology which is used for determining all the possible relations that can be identified among
concepts. 
The possible relations are weighted  using a GNN and structural information of the graph.

## Table of Contents

* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)



## General Information
This repository contains the source code behind the work:
"A semantic approach for constructing knowledge graphs  from web tables" authored by
S.Bonfitto, M. DiLeo, S. Gaito, E. Casiraghi, M. Mesiti.

The approach uses a GNN architecture for prediction tasks on knowledge graphs. 
Specifically, we solved multi-relational link prediction and property detection (i.e. link classification).

### Description of the model
We developed a model named __HeteroGNN__. In our model we defined two hetero convolutional layers (HeteroConv) with graph attention mechanism to compute the embeddings. Each relation name has its attention mechanism and the embedding of a node is obtained as the sum of the contributions of each convolution defined on the relations in which it is involved. The graph attention mechanism has been introduced since many real-world KGs tend to contain relationships from multiple sources of varying quality (e.g. interactions extracted from unstructured text are less reliable than manually curated ones).  We considered both concepts and properties. We used a constant value 1 as a node feature for each concept to capture its structural information. Whereas for properties, we treat them as nodes and we compute specific features for each property type. We used DistMult as decoder function.

We compared our model with two others GNN architectures: SeMi and MRGCN. The following table summarizes the main characteristics of the models.

| method         | GNN layer | node features | nodes         | edges         |
|----------------|-----------|---------------|---------------|---------------|
| HeteroGNN      | HeteroGAT | yes           | heterogeneous | heterogeneous |
| MRGCN          | R-GCN     | yes           | homogeneous   | heterogeneous |
| SeMi           | R-GCN     | no            | homogeneous   | heterogeneous |

### Keypoints of our work
- We developed a model that can handle both heterogeneous nodes and edges and introduces a graph attention mechanism for varying quality relationships.
- We showed that our model outperforms other two GNN models used for multi-relational link prediction task in KGs on three real-world heterogeneous graph datasets.
- We investigated the feasibility of solving property detection, a link classification task in which we assign the correct label to a link, that exists for sure, between an entity and a property in a KG. We evaluated the prediction performance of our model on this task using a link prediction setting. The results confirms the feasibility of our approach for the property detection task.


## Technologies Used
- torch-geometric - version 2.2.0
- rdflib - version 6.2.0


## Datasets
The ontology and KG files related to Movie, Area, and Public Procurements (PP) datasets are available to download [here]().


## Setup
The dependencies of this project are listed in requirements.txt 
We used a python3 environment


## Usage
The AUROCs shown in the article can be reproduced through the jupyter notebooks.
The full approach can be executed using the main.py
The config.ini file should be checked for changing the dataset.

`write-your-code-here`
