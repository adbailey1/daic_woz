#! /usr/bin/env bash

FILES_PATH='/home/andrew/PycharmProjects/daic_woz_att_raw'
#FILES_PATH='/user/HS227/ab01814/pycharm_projects/daic_woz_att'
#WITH GPU
python3.7 $FILES_PATH/main.py train --validate --vis --cuda
#WITHOUT GPU
# python3.7 $FILES_PATH/main.py train --validate --vis



