# APBM
Augment physics-based models

Code for the IEEE TAES paper:  

T. Imbiriba, O. Straka, J. Duník and P. Closas, "Augmented physics-based machine learning for navigation and tracking," in IEEE Transactions on Aerospace and Electronic Systems, doi: 10.1109/TAES.2023.3328853.

https://ieeexplore.ieee.org/abstract/document/10302385

Run pvtdata_sim.m for the experiments with real data. 

Run synthetic_tracking_example_mc.m for the simulated example.


Data-driven models have great potential to characterize complex systems. However, physics-based models are extremely important as they are able to summarize tons of information from experiments, solid theory, experts, and common sense. Also, physics-based models typically come with easily interpretable parameters that have physical meaning. Why throw that knowledge away when training data-driven models? our proposed augmented physics-based models (APBMs) leverage both worlds. The TAES article trades off both modeling options and presents promising solutions in between, whereby physics-based models are augmented by data-driven components.
