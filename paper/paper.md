---
title: 'MOTrainer: Distributed Measurement Operator Trainer for Data Assimilation Applications'
tags:
  - Python
  - Measurement Operator
  - Data Assimilation
  - Machine Learning
  - Kalman Filter
authors:
  - name: Ou Ku
    orcid: 0000-0002-5373-5209
    affiliation: 1 
  - name: Fakhereh Alidoost
    affiliation: 1
  - name: Xu Shan
    affiliation: 2
  - name: Pranav Chandramouli
    affiliation: 1
  - name: Sonja Georgievska
    affiliation: 1
  - name: Meiert W. Grootes
    affiliation: 1 
  - name: Susan Steele-Dunne
    corresponding: true
    affiliation: 2
affiliations:
 - name: Netherlands eScience Center, Netherlands
   index: 1
 - name: Delft University of Technology, Netherlands
   index: 2
date: 22 Dec 2023
bibliography: paper.bib
---

## Summary

Data Assimilation (DA) remains a pivotal data analytical technique in environmental science research, enabling the constraint of physical model states with observation data.

In the DA process, observation data is integrated into the physical model through the application of a Measurement Operator (MO) – a connection model mapping physical model states to observations. Researchers have observed that employing a Machine-Learning (ML) model as a surrogate MO can bypass the limitations associated with using an overly simplified MO [@Forman:2014; @XUE:2015; @Forman:2017].

## Statement of Need

ML models should undergo training within a coherent spatio-temporal scope, where physical model states can be consistently mapped to observations using the same model. Dealing with a large spatio-temporal scale may involve multiple mapping processes, necessitating the consideration of training separate models for distinct spatial and/or temporal partitions of the dataset. As the number of partitions increases, a challenge emerges in effectively distributing these training tasks among the partitions.

A surrogate MO, as a ML model, should be trained over a coherent spatio-temporal scope, where one can assume that the same MO applies when mapping physical model states to observations. When dealing with a large spatio-temporal scale, multiple mapping processes may exist, prompting consideration for training separate MOs for distinct spatial and/or temporal partitions of the dataset. As the number of partitions increases, a challenge arises in distributing these training tasks effectively among the partitions.

To address this challenge, we developed the open Python package `MOTrainer`. It caters to researchers requiring training independent MOs across extensive spatio-temporal coverage in a distributed manner. `MOTrainer` leverages Xarray's support for multi-dimensional datasets to accommodate spatio-temporal features of input/output data of the training tasks. It provides user-friendly functionalities implemented with the Dask library, facilitating the partitioning of large spatio-temporal data for independent model training tasks. Additionally, it streamlines the train-test data split based on customized spatio-temporal coordinates. The Jackknife method [@mccuen1998hydrologic] is implemented as an external Cross-Validation (CV) method for Deep Neural Network (DNN) training, with support for Dask parallelization. This feature enables the scaling of training tasks across various computational infrastructures.

`MOTrainer` has been employed in a study of vegetation water dynamics [@shan:2022], where it facilitated the mapping of Land-Scape Model (LSM) states to satellite radar observations.

## Tutorial

The `MOTrainer` package includes comprehensive [usage examples](https://vegewaterdynamics.github.io/motrainer/usage_split/), as well as tutorials for:

1. Converting input data to Xarray Dataset format: [Example 1](https://vegewaterdynamics.github.io/motrainer/notebooks/example_read_from_one_df/) and [Example 2](https://vegewaterdynamics.github.io/motrainer/notebooks/example_read_from_one_df/);

2. Training tasks on simpler ML models using `sklearn` and `daskml`: [Example Notebook](https://vegewaterdynamics.github.io/motrainer/notebooks/example_daskml/);

3. Training tasks on Deep Neural Networks (DNN) using TensorFlow: [Example Notebook](https://vegewaterdynamics.github.io/motrainer/notebooks/example_dnn/).

## Acknowledgements

The authors express sincere gratitude to the Dutch Research Council (Nederlandse Organisatie voor Wetenschappelijk Onderzoek, NWO) and the Netherlands Space Office for their generous funding of the MOTrainer development through the User Support Programme Space Research (GO) call, grant ALWGO.2018.036. Special thanks to SURF for providing valuable computational resources for MOTrainer testing via the grant EINF-339.

We would also like to thanks Dr. Francesco Nattino, Dr. Yifat Dzigan, Dr. Paco López-Dekker, and Tina Nikaein for the insightful discussions, which are important contributions to this work.

## References