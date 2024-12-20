# APBM
Augment physics-based models

Code for the simulations in IEEE TAES paper:  

Imbiriba, T., Straka, O., Duník, J. and Closas, P., 2023. Augmented physics-based machine learning for navigation and tracking. IEEE Transactions on Aerospace and Electronic Systems.

https://ieeexplore.ieee.org/abstract/document/10302385


More details about APBM can also be found in:

Imbiriba, T., Demirkaya, A., Duník, J., Straka, O., Erdoğmuş, D. and Closas, P., 2022, July. Hybrid neural network augmented physics-based models for nonlinear filtering. In 2022 25th International Conference on Information Fusion (FUSION) (pp. 1-6). IEEE.

https://arxiv.org/pdf/2204.06471

Run pvtdata_sim.m for the experiments with real data. 

Run synthetic_tracking_example_mc.m for the simulated example.


Data-driven models have great potential to characterize complex systems. However, physics-based models are extremely important as they are able to summarize tons of information from experiments, solid theory, experts, and common sense. Also, physics-based models typically come with easily interpretable parameters that have physical meaning. Why throw that knowledge away when training data-driven models? our proposed augmented physics-based models (APBMs) leverage both worlds. The TAES article trades off both modeling options and presents promising solutions in between, whereby physics-based models are augmented by data-driven components.
