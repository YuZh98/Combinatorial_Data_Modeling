# Combinatorial Data Modeling

This repository accompanies the paper  
**"Statistical Modeling of Combinatorial Response Data"**  
and contains R and C++ code used for simulation studies presented in the paper.

---

## 🧠 Overview

In categorical data analysis, conventional models for binary or multinomial outcomes often fail when responses are **combinatorial**—i.e., integer-valued arrays subject to additional structural constraints. Such response types arise naturally in settings like:
- Surveys with skip logic
- Event propagation on networks
- Observed matching in ecological systems

Standard generalized linear models do not capture the feasible region of these structured responses, potentially leading to biased inference.

This repository implements a Bayesian approach that:
- Treats the observed response as a **deterministic transformation** of a continuous latent variable.
- Defines this transformation as the **solution to an Integer Linear Program (ILP)**.
- Uses a **Gibbs sampler with a custom hit-and-run step** to perform efficient posterior inference.

This framework generalizes the classical **probit data augmentation** approach to the combinatorial setting.

---

## 📁 Repository Contents

```bash
.
├── reg_simulation.R      # Main script: data generation, Gibbs sampling, post-processing
├── hit_and_run.cpp       # C++: custom constrained hit-and-run sampler
├── tum_check.cpp         # C++: check total unimodularity (TUM) of constraint matrix
└── README.md             # Project documentation
