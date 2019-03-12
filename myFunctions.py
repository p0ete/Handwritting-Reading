import cv2
import numpy as np
import os
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks
import datetime

import random 

def uniqueid():
    seed = random.getrandbits(32)
    while True:
       yield seed
       seed += 1

def order_rects(predicted_boxes):
	
	original_rects = []
	rects = []
	lines = []
	data_words = []
	for box in predicted_boxes:
		x1, y1 = box[0]
		x2, y2 = box[1]
		x3, y3 = box[2]
		x4, y4 = box[3]
		
		x = int(min(x1,x4))
		y = int(min(y1,y2))
		w = int(max(x2,x3) - x)
		h = int(max(y3,y4) - y)
		original_rects.append([x,y,w,h])
	
	h_mean= np.mean([r[-1] for r in original_rects])
	
	for r in original_rects:
		x,y,w,h = r
		if len(lines) == 0:
			rects.append([[x,y,w,h]])
			lines.append([y+(h/2)])
		else:
			find = False
			for j in range(len(lines)):
				if y+(h/2) == np.mean(lines[j]) or \
				(y+(h/2) >= np.mean(lines[j]) - (h_mean) and y+(h/2) <= np.mean(lines[j])+  (h_mean)):
					lines[j].append(y+(h/2))
					rects[j].append([x,y,w,h])
					find = True
					break
					
			if not find:
				rects.append([[x,y,w,h]])
				lines.append([y+(h/2)])
	
	tmp = []
	for line in lines:
		tmp.append(np.mean(line))
	
	new_rects= [rects[i] for i in np.argsort(tmp)]        
	
	rects = []
	for line in new_rects:
		tmp = []
		for word in line:
			x,y,w,h = word
			tmp.append(x + (w/2))
		
		new_line = [line[i] for i in np.argsort(tmp)]
		rects.append(new_line)
	
	return rects

def uniforming_brightness(image, mean_cible): 
	if np.max(image) <=1:
		normalized = True
	else:
		normalized = False

	mean = np.mean(image)

	if normalized:
		if mean < mean_cible:
			image += mean_cible - mean
		else:
			image -= mean - mean_cible
		if np.max(image) > 1 or np.min(image) < 0:	
			for i in range(image.shape[0]):
				for j in range(image.shape[1]):
					image[i][j] = min(max(0., image[i][j]), 1.)
	else:
		if mean < mean_cible:
			lim = 255 - int(mean_cible - mean)
			image[image >= lim] = 255
			image[image < lim] += int(mean_cible - mean)
		else:
			lim = int(mean-mean_cible)
			image[image<=lim]= 0
			image[image>lim] -= int(mean - mean_cible)
		# print(np.max(image))
		# if np.max(image) > 255 or np.min(image) < 0:
		# 	for i in range(image.shape[0]):
		# 		for j in range(image.shape[1]):
		# 			image[i][j] = min(max(0, image[i][j]), 255)

				
	return image

def normalize(image):
	image = image/255
	image = np.float32(image)
	return image

def draw_rects(image, rectangles, rectangles_labels = [], wait_key = True,save_folder = ""):
	image_rect = image.copy()

	
	if len(rectangles[0]) != 4:
		tmp = []
		for line in rectangles:
			for rect in line:
				tmp.append(rect)
		rectangles = tmp	        

	if image_rect.shape[-1] != 3:
		image_rect = cv2.cvtColor(image_rect,cv2.COLOR_GRAY2RGB)
	for idx in range(len(rectangles_labels)):
		x, y, w, h = rectangles_labels[idx]
		cv2.rectangle(image_rect, (x, y), (x+w-1, y+h-1), (255, 0, 0), 2)

	for idx in range(len(rectangles)):
		x, y, w, h = rectangles[idx]
		cv2.rectangle(image_rect, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)

	if not wait_key:
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(image_rect,'Click on "q" to close the window.' ,(10,50), font, 1,(0,0,0),1,cv2.LINE_AA)
	if len(save_folder) == 0:
		cv2.imshow('', image_rect)
		if wait_key:
			cv2.waitKey(0)
			cv2.destroyAllWindows() 
			
	else:
		#image_rect = image_rect*255
		cv2.imwrite(save_folder,image_rect)

def resize_image(image, rects,x0,y0,new_w, new_h, with_border_rect = True):
	## ---(x_min, y_min).###########---------------
	## ------------------#---------#---------------
	## ------------------#---------#---------------
	## ------------------###########.(x_max, y_max)
	# x_min = 20
	# y_min = 20
	x_max = x0 + new_w
	y_max = y0 + new_h
	
	height, width = image.shape
	if x_max > width:
		print("The new width is higher thant the previous one !")
	if y_max > height:
		print("The new height is higher thant the previous one !")
	
	imS = image[y0:y_max, x0:x_max]
	rectS = []
	for rect in rects:
		x1, y1, w, h = rect
		x1 = int(x1) - x0
		y1 = int(y1) - y0
		w = int(w)
		h = int(h)
		x2 = x1+w
		y2 = y1 + h
		
		if x2 >= x0 and x1 < x_max and y2 >= y0-h and y1 < y_max-h:
			new_x1 = x1
			new_y1 = y1
			new_x2 = x2
			new_y2 = y2
			if with_border_rect:
				
				if new_x1< 0:
					new_x1 = 0
				if new_y1 < 0:
					new_y1 = 0
				if new_x2 > (x_max - x0):
					new_x2 = x_max - x0 -1 
				if new_y2 > (y_max -  y0):
					new_y2 = y_max - y0 - 1
				rectS.append([new_x1, new_y1, new_x2-new_x1, new_y2-new_y1])
			else:
				if new_x1>= 0 and new_y1 >= 0 and new_x2 <= (x_max - x0) and new_y2 <= (y_max -  y0):
					rectS.append([new_x1, new_y1, new_x2-new_x1, new_y2-new_y1])
	
	return imS, rectS

def rotate_image(small, rectangles, angle):
	image = small.copy()
	rows,cols = image.shape
	(height, width) = image.shape[:2]
	(cX, cY) = (width // 2, height // 2)

	M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])

	# compute the new bounding dimensions of the image
	nW = int(round((height * sin) + (width * cos)))
	nH = int(round((height * cos) + (width * sin)))

	M[0, 2] += (nW / 2) - cX
	M[1, 2] += (nH / 2) - cY

	dst = cv2.warpAffine(image,M,(nW,nH), cv2.BORDER_CONSTANT, borderValue= np.mean(image))
	corners = []
	for idx in range(len(rectangles)):
		x, y, w, h = rectangles[idx]
		x1 = x
		y1 = y

		x2 = x1 + w
		y2 = y1 

		x3 = x1
		y3 = y1 + h

		x4 = x2
		y4 = y3

		corners.append([x1,y1,x2,y2,x3,y3,x4,y4])

	corners_np = np.array(corners)
	corners_np = corners_np.reshape(-1,2)
	corners_np = np.hstack((corners_np, np.ones((corners_np.shape[0],1), dtype = type(corners_np[0][0]))))
	calculated = np.dot(M,corners_np.T).T
	calculated = calculated.reshape(-1,8)
	
	calculated_rects = []
	for idx in range(len(calculated)):
		x1,y1,x2,y2,x3,y3,x4,y4 = calculated[idx]
		x = x1
		#y = y1 + (-np.sign(angle) * (height/(abs(angle)*900)))
		y = y1
		w = x2- x1
		h = y3-y1

		h = h+ (abs(angle)*height/(1000))
		#w = w+ (width/1000)

		calculated_rects.append([int(round(x)), int(round(y)), int(round(w)), int(round(h))])
	
	return dst, calculated_rects

def squeeze_image(image,rectangles, width_squeeze_coef, height_squeeze_coef):
	#height_squeeze_coef = 1.5
	#width_squeeze_coef = 1

	(height, width) = image.shape[:2]
	res = cv2.resize(image,(int(width_squeeze_coef*width), int(height_squeeze_coef*height)), interpolation = cv2.INTER_CUBIC)

	calculated_rects = []
	for idx in range(len(rectangles)):

		x,y, w,h = rectangles[idx]

		if width_squeeze_coef >= 1:
			x1 = x * width_squeeze_coef
			w1 = w * width_squeeze_coef
		else:
			x1 = x * width_squeeze_coef
			w1 = w * width_squeeze_coef

		if height_squeeze_coef >= 1:
			y1 = y * height_squeeze_coef
			h1 = h * height_squeeze_coef
		else:
			y1 = y * height_squeeze_coef
			h1 = h * height_squeeze_coef
		calculated_rects.append([int(round(x1)),int(round(y1)),int(round(w1)),int(round(h1))])
	return res, calculated_rects

def pad_image(image, rect, desired_w, desired_h):
	new_rect = []
	height, width = image.shape
	top = 0
	bottom = 0
	left = 0
	right = 0
	if height < desired_h:
		if (desired_h-height) % 2 == 0:
			top = (desired_h-height) //2
			bottom = (desired_h-height) //2
		else:
			top = int((desired_h-height) /2)
			bottom = int((desired_h-height) /2)+1
		
	if width < desired_w:
		if (desired_w-width) % 2 == 0:
			left = (desired_w-width) //2
			right = (desired_w-width) //2
		else:
			left = int((desired_w-width) /2)
			right = int((desired_w-width) /2)+1
	
	image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value= np.mean(image))
	for r in rect:
		x, y, w, h = r
		new_r = [x+left, y+ top, w, h]
		new_rect.append(new_r)
	return image, new_rect

