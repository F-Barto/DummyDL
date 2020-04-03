#! /bin/bash
PARAMS_TO_TEST="batch_size channels metric learning_rate decay_rate"
python ../../config_utils.py ./test.yml default $PARAMS_TO_TEST
echo "---------------------------------"
PARAMS_TO_TEST="batch_size additional_param"
python ../../config_utils.py ./test.yml large $PARAMS_TO_TEST