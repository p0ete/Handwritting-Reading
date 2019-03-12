import os
import cv2
import matplotlib.pyplot as plt    
import json
import math
import numpy as np
from myFunctions import *
import random
from CRNN.myModel import *

from keras.models import model_from_json
from keras import backend as K
import tensorflow as tf
import gc

from EAST.for_detect import *

import time

CRNN_weights = "./weights_crnn.h5"

EAST_model_file ='./model_east.json'
EAST_weighst = "./weights_east.h5"

def predict_boxes(whole_image):

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.Session(config=config)
	
	RESIZE_FACTOR= 2

	# load trained model
	json_file = open(EAST_model_file, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	EAST_model = model_from_json(loaded_model_json, custom_objects={'tf': tf, 'RESIZE_FACTOR': RESIZE_FACTOR})
	EAST_model.load_weights(EAST_weighst)

	img_resized, (ratio_h, ratio_w) = resize_image(whole_image)
	img_resized = (img_resized / 127.5) - 1

	# feed image into model
	score_map, geo_map = EAST_model.predict(img_resized[np.newaxis, :, :, :])
	boxes = detect(score_map=score_map, geo_map=geo_map)

	if boxes is not None:
		boxes = boxes[:, :8].reshape((-1, 4, 2))
		boxes[:, :, 0] /= ratio_w
		boxes[:, :, 1] /= ratio_h

	rects = order_rects(boxes)
	K.clear_session()

	return rects

def translate(whole_image, rects):
	CRNN_model, y_pred, inputs = get_Model(False)
	CRNN_model.load_weights(CRNN_weights)

	whole_image_gray = cv2.cvtColor(whole_image,cv2.COLOR_RGB2GRAY)


	text = []
	for line in rects:
		translation = ""
		for rect in line:
			x,y,w,h = rect
			word_image = whole_image_gray[y:(y+h), x:(x+w)]
			#data_words.append(word_image)
			new_image, _ = pad_image(word_image, [], WIDTH, HEIGHT)
			translation += predict(CRNN_model, new_image) + " "
		text.append(translation)# += "\n"

	return text