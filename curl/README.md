# CURL

This is based on the official implementation of [CURL](https://mishalaskin.github.io/curl/) for the DeepMind control suite experiments.

## Installation 

All of the dependencies are in the `conda_env.yml` file, and can be installed with the following command:

```
conda env create -f conda_env.yml
```

## Instructions
To train a CURL agent on the `cartpole swingup` task from image-based observations run `bash script/run.sh` from the root of this directory. The `run.sh` file contains the following command, which you can modify to try different environments / hyperparamters.
```
CUDA_VISIBLE_DEVICES=0 python train.py \
    --domain_name cartpole \
    --task_name swingup \
    --encoder_type pixel \
    --action_repeat 8 \
    --save_tb --pre_transform_image_size 100 --image_size 84 \
    --work_dir ./tmp \
    --agent curl_sac --frame_stack 3 \
    --seed -1 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 --num_train_steps 1000000 
```

