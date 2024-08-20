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
    orcid: 0000-0001-8407-6472
    affiliation: 1
  - name: Xu Shan
    orcid: 0000-0002-0569-4326
    affiliation: 2
  - name: Pranav Chandramouli
    orcid: 0000-0002-7896-2969
    affiliation: 1
  - name: Sonja Georgievska
    orcid: 0000-0002-8094-4532
    affiliation: 1
  - name: Meiert W. Grootes
    orcid: 0000-0002-5733-4795
    affiliation: 1 
  - name: Susan Steele-Dunne
    orcid: 0000-0002-8644-3077
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

Data assimilation (DA) is an essential procedure in Earth and environmental sciences, enabling physical model states to be constrained using observational data. [@REICHLE20081411; @Evensen2009; @Clement2018; @Carrassi2018]

In the DA process, observations are integrated into the physical model through the application of a Measurement Operator (MO) – a connection model mapping physical model states to observations. Researchers have observed that employing a Machine-Learning (ML) model as a surrogate MO can bypass the limitations associated with using an overly simplified MO. [@Forman:2014; @XUE:2015; @Forman:2017]

## Statement of Need

The surrogate MO, trained as a ML model, is generally considered valid within a specific spatio-temporal range. [@zhou2008ensemble; @REICHLE20081411; @shan:2022] When dealing with a large spatio-temporal scale, multiple mapping processes may exist, prompting consideration for training separate MOs for distinct spatial and/or temporal partitions of the dataset. As the number of partitions increases, a challenge arises in distributing these training tasks effectively among the partitions.

To address this challenge, we developed a novel approach for distributed training of MOs. We present the open Python library `MOTrainer`, which to the best of our knowledge, is the first Python library catering to researchers requiring training independent MOs across extensive spatio-temporal coverage in a distributed manner. `MOTrainer` leverages Xarray's [@Hoyer_xarray_N-D_labeled_2017] support for multi-dimensional datasets to accommodate spatio-temporal features of input/output data of the training tasks. It provides user-friendly functionalities implemented with the Dask [@Rocklin2015DaskPC] library, facilitating the partitioning of large spatio-temporal data for independent model training tasks. Additionally, it streamlines the train-test data split based on customized spatio-temporal coordinates. The Jackknife method [@mccuen1998hydrologic] is implemented as an external Cross-Validation method for Deep Neural Network (DNN) training, with support for Dask parallelization. This feature enables the scaling of training tasks across various computational infrastructures.

`MOTrainer` has been employed in a study of vegetation water dynamics [@shan:2022], where it facilitated the mapping of Land-Scape Model states to satellite radar observations.

## Tutorial

The `MOTrainer` package includes comprehensive [usage examples](https://vegewaterdynamics.github.io/motrainer/usage_split/), as well as tutorials for:

1. Converting input data to Xarray Dataset format: [Example 1](https://vegewaterdynamics.github.io/motrainer/notebooks/example_read_from_one_df/) and [Example 2](https://vegewaterdynamics.github.io/motrainer/notebooks/example_read_from_one_df/);

2. Training tasks on simpler ML models using `sklearn` and `daskml`: [Example Notebook](https://vegewaterdynamics.github.io/motrainer/notebooks/example_daskml/);

3. Training tasks on Deep Neural Networks (DNN) using TensorFlow: [Example Notebook](https://vegewaterdynamics.github.io/motrainer/notebooks/example_dnn/).

## Acknowledgements

The authors express sincere gratitude to the Dutch Research Council (Nederlandse Organisatie voor Wetenschappelijk Onderzoek, NWO) and the Netherlands Space Office for their generous funding of the MOTrainer development through the User Support Programme Space Research (GO) call, grant ALWGO.2018.036. Special thanks to SURF for providing valuable computational resources for MOTrainer testing via the grant EINF-339.

We would also like to thanks Dr. Francesco Nattino, Dr. Yifat Dzigan, Dr. Paco López-Dekker, and Tina Nikaein for the insightful discussions, which are important contributions to this work.

## References
