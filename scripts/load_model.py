import tensorflow as tf
import numpy as np
from GSGM import GSGM
import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config_cluster.json', help='Training parameters')
parser.add_argument('--factor', type=int,default=1, help='Step reduction for distillation model')
flags = parser.parse_args()

npart = 200
model_name = 'GSGM'
config = utils.LoadJson(flags.config)
model = GSGM(config=config,factor=flags.factor,npart=npart)
config = utils.LoadJson(flags.config)
checkpoint_folder = '../checkpoints_{}/checkpoint'.format(model_name)
model.load_weights('{}'.format(checkpoint_folder)).expect_partial()
print("\nMODEL TYPE = ",type(model))

model.build(input_shape=(2,))
model.summary()
print("Done???")
