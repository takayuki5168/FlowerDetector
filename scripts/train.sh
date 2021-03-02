#!/bin/sh

SCRIPT_DIR=$(cd $(dirname $0); pwd)

. $SCRIPT_DIR/../venv/bin/activate

# VGG
python $SCRIPT_DIR/trainer.py -g 0 -o SGD -l 0.005 -w 0.0005 &
python $SCRIPT_DIR/trainer.py -g 1 -o Adam -l 0.00001 -w 0 &
python $SCRIPT_DIR/trainer.py -g 2 -o Adam -l 0.00001 -w 0.0001 &
python $SCRIPT_DIR/trainer.py -g 3 -o Adam -l 0.000005 -w 0.0001 &

# MobileNet
#python $SCRIPT_DIR/trainer.py -g 0 -o SGD -l 0.05 -w 0.0005 &
#python $SCRIPT_DIR/trainer.py -g 1 -o Adam -l 0.0001 -w 0 &
#python $SCRIPT_DIR/trainer.py -g 2 -o Adam -l 0.0001 -w 0.0001 &
#python $SCRIPT_DIR/trainer.py -g 3 -o Adam -l 0.001 -w 0.0001 &


sleep 10
tensorboard --logdir=$SCRIPT_DIR"/../logs/"
