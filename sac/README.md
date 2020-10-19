# SAC-D2RL


Need to install mujoco_py https://github.com/openai/mujoco-py, mujoco200 (license required), [OpenAI gym](https://github.com/openai/gym), [DeepMind Control Suite](https://github.com/deepmind/dm_control), and [RealWorld RL Environments](https://github.com/google-research/realworldrl_suite)


### Installation

Dependencies are listed in `dependencies.yml` and can be installed with

`conda env create -f dependencies.yml`



### Usage

To run experiments with SAC-D2RL, say on the OpenAI Gym Ant-v2 environment, 

`python3 main.py --env ant`

For other environments, check the configs in `main.py`
