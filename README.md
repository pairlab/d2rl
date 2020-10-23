# D2RL
Official PyTorch code for D2RL: Deep Dense Architectures in Reinforcement Learning. Details on an independently reproduced ***TF2*** implementation is listed below.

***Paper***: http://arxiv.org/abs/2010.09163

***Blog***: https://sites.google.com/view/d2rl/home

The code includes the code to train SAC-D2RL, TD3-D2RL, and CURL-D2RL agents. 

***If there are any issues or questions related to the code, send an email at: samarth.sinha@mail.utoronto.ca***

To try D2RL on other environments, the main parameters to be tuned are the learning rate of the actors and critics. To try D2RL with other algorithms, we also include a pseudo-code for the architecture changes in the main paper. Kindly let us know if there are any questions.

Installation details, dependencies, and instructions for training are included in the individual sub-folders. 

***TF2 version of the D2RL which was independently reproduced:***

https://github.com/keiohta/tf2rl

More details on D2RL and how to run the code can be found in

- https://github.com/keiohta/tf2rl/blob/master/examples/run_d2rl_sac.py

- https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/d2rl_sac.py


## Acknowledgement

The codebase is built upon other these previous codebases: 

SAC: https://github.com/denisyarats/pytorch_sac

TD3: https://github.com/sfujim/TD3

CURL: https://github.com/MishaLaskin/curl 
