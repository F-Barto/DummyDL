#! /bin/bash

CONFIG_DIR="../configs/"

python ../train.py \
--gpus '0' \
--model_config_file $CONFIG_DIR"simplemnist.yml" \
--model_config_profile "default" \
--project_config_file $CONFIG_DIR"project.yml" \
--project_config_profile "default"
