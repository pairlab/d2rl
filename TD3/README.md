# TD3-D2RL

This is based on the PyTorch implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3). [paper](https://arxiv.org/abs/1802.09477).

Need to install mujoco_py https://github.com/openai/mujoco-py, mujoco200 (license required), and [OpenAI gym](https://github.com/openai/gym)


### Usage

To run experiments with TD3-D2RL, say on the OpenAI Gym Ant-v2 environment, 

`python3 main.py --env ant`

For other environments, check the configs in `main.py`
