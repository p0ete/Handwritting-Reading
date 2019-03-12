from keras.applications import VGG16
from keras.layers import Reshape, Lambda, Input, Activation
from keras.models import Sequential
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.callbacks
import tensorflow as tf
from keras import backend as K
import editdistance
import gc

import random
import os
import numpy as np
import cv2
from keras.optimizers import Adadelta
import datetime
import pylab

import itertools

alphabet = u"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,;:!?'-çàâäéèêëïîôûüùœÀÂÉÈÊÔ0123456789«»()"
OUTPUT_SIZE = len(alphabet) + 1

# Network parameters
ACT = 'relu'
OUTPUT_DIR = './CRNN_output/'

CONV_FILTERS = 16
KERNEL_SIZE = (3, 3)
POOL_SIZE = 2
TIME_DENSE_SIZE = 32
RNN_SIZE = 512
BATCH_SIZE = 64
HEIGHT = 75
WIDTH =256
DOWNSAMPLE_FACTOR = 4
ABSOLUTE_MAX_STRING_LENGTH = 20
input_shape = (HEIGHT, WIDTH,3)

NB_MAX_IMAGES = 16000

FOLDER_IMG ='../data/train/words_images/'
EXTENSION ='.png'
LABEL_FILE = '../data/train/words_images/labels.txt'

def check_alphabet(actual_alphabet, labels):

    a = [word for word in labels.values()]
    b = list(set(a))
    
    t = ""
    for word in b:
        t+=word
    
    alphabet2 = ""
    for letter in list(set(t)):
        alphabet2 += letter
    
    letters_to_add = ""
    for letter in alphabet2:
        if letter not in actual_alphabet:
            print(letter)
            letters_to_add+=letter
    return letters_to_add
    
def predict(model, img):
	img = cv2.resize(img, (WIDTH, HEIGHT))
	img = img.astype(np.float32)
	img = (img / 255.0) * 2.0 - 1.0
	img = img.T
	img = np.expand_dims(img, -1)
	img = np.expand_dims(img, axis=0)

	prediction = model.predict(img)

	out = prediction
	ret = []
	for j in range(out.shape[0]):
		out_best = list(np.argmax(out[j, 2:], 1))
		out_best = [k for k, g in itertools.groupby(out_best)]
		outstr = labels_to_text(out_best)
		ret.append(outstr)
	return ret[0]

# # Input data generator
def labels_to_text(labels):
	alphabet_bl = alphabet + " "
	ret = []
	for c in labels:
		if c == (len(alphabet_bl)-1):  # CTC Blank
			ret.append("")
		else:
			ret.append(alphabet_bl[c])
	return "".join(ret)

def text_to_labels(text):
	alphabet_bl = alphabet + " "
	return list(map(lambda x: alphabet.index(x), text))

def decode_batch(test_func, word_batch):
	out = test_func([word_batch])[0]
	ret = []
	for j in range(out.shape[0]):
		out_best = list(np.argmax(out[j, 2:], 1))
		out_best = [k for k, g in itertools.groupby(out_best)]
		outstr = labels_to_text(out_best)
		ret.append(outstr)
	return ret

def ctc_lambda_func(args):
	y_pred, labels, input_length, label_length = args
	# the 2 is critical here since the first couple outputs of the RNN
	# tend to be garbage:
	y_pred = y_pred[:, 2:, :]
	return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_Model(training):
	#WIDTH, HEIGHT = 128, 64
	input_shape = (WIDTH, HEIGHT, 1)     # (128, 64, 1)

	# Make Networkw
	inputs = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 128, 64, 1)

	# Convolution layer (VGG)
	inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(inputs)  # (None, 128, 64, 64)
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)
	inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,64, 32, 64)

	inner = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)  # (None, 64, 32, 128)
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)
	inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, 32, 16, 128)

	inner = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 256)
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)
	inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 256)
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)
	inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)  # (None, 32, 8, 256)

	inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(inner)  # (None, 32, 8, 512)
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)
	inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)  # (None, 32, 8, 512)
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)
	inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)  # (None, 32, 4, 512)

	inner = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(inner)  # (None, 32, 4, 512)
	inner = BatchNormalization()(inner)
	inner = Activation('relu')(inner)

	# CNN to RNN
	inner = Reshape(target_shape=((64, 2048)), name='reshape')(inner)  # (None, 32, 2048)
	
	inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 32, 64)

	# RNN layer
	lstm_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)  # (None, 32, 512)
	lstm_1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
	lstm1_merged = add([lstm_1, lstm_1b])  # (None, 32, 512)
	lstm1_merged = BatchNormalization()(lstm1_merged)
	lstm_2 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
	lstm_2b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(lstm1_merged)
	lstm2_merged = concatenate([lstm_2, lstm_2b])  # (None, 32, 1024)
	lstm_merged = BatchNormalization()(lstm2_merged)

	# transforms RNN output to character activations:
	inner = Dense(OUTPUT_SIZE, kernel_initializer='he_normal',name='dense2')(lstm2_merged) #(None, 32, 63)
	y_pred = Activation('softmax', name='softmax')(inner)

	labels = Input(name='the_labels', shape=[ABSOLUTE_MAX_STRING_LENGTH], dtype='float32') # (None ,8)
	input_length = Input(name='input_length', shape=[1], dtype='int64')     # (None, 1)
	label_length = Input(name='label_length', shape=[1], dtype='int64')     # (None, 1)

	# Keras doesn't currently support loss funcs with extra parameters
	# so CTC loss is implemented in a lambda layer
	loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length]) #(None, 1)

	if training:
		return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out), y_pred, inputs
	else:
		return Model(inputs=[inputs], outputs=y_pred), y_pred, inputs

class TextImageGenerator:
	def __init__(self, id_list, my_labs, img_w, img_h,
				 batch_size, downsample_factor, nb_max_images, max_text_len=ABSOLUTE_MAX_STRING_LENGTH):
		self.img_h = img_h
		self.img_w = img_w
		self.labels = my_labs
		self.batch_size = batch_size
		self.max_text_len = max_text_len
		self.downsample_factor = downsample_factor
		#self.img_dirpath = img_dirpath                  # image dir path
		#self.img_dir = os.listdir(self.img_dirpath)     # images list
		
		self.id_list = id_list
		random.shuffle(self.id_list)
		if len(self.id_list) > nb_max_images:
			self.id_list = random.sample(self.id_list, nb_max_images)
		
		self.id_list_train = self.id_list[0:int(len(self.id_list)*0.8)]
		self.id_list_val = self.id_list[int(len(self.id_list)*0.8):]
		
		self.n_train = len(self.id_list_train)                      # number of images
		self.indexes_train = list(range(self.n_train))
		self.cur_index_train = 0
		self.imgs_train = np.zeros((self.n_train, self.img_h, self.img_w))
		self.texts_train = []
		
		self.n_val = len(self.id_list_val)                      # number of images
		self.indexes_val = list(range(self.n_val))
		self.cur_index_val = 0
		self.imgs_val = np.zeros((self.n_val, self.img_h, self.img_w))
		self.texts_val = []
		
		self.blank_label = len(alphabet) + 1

	def build_data(self):
		print(self.n_train, " Image Loading start...")
		for i, ID in enumerate(self.id_list_train):
						
			#img = cv2.imread(self.img_dirpath + img_file, cv2.IMREAD_GRAYSCALE)
			img = cv2.imread(FOLDER_IMG + ID + EXTENSION, cv2.IMREAD_GRAYSCALE)
			
			img = cv2.resize(img, (self.img_w, self.img_h))
			img = img.astype(np.float32)
			img = (img / 255.0) * 2.0 - 1.0

			self.imgs_train[i, :, :] = img
			self.texts_train.append(self.labels[ID])
		
		for i, ID in enumerate(self.id_list_val):
						
			#img = cv2.imread(self.img_dirpath + img_file, cv2.IMREAD_GRAYSCALE)
			img = cv2.imread(FOLDER_IMG + ID + EXTENSION, cv2.IMREAD_GRAYSCALE)
			
			img = cv2.resize(img, (self.img_w, self.img_h))
			img = img.astype(np.float32)
			img = (img / 255.0) * 2.0 - 1.0

			self.imgs_val[i, :, :] = img
			self.texts_val.append(self.labels[ID])
			
		#print(len(self.texts_train) == self.n_train)
		print(" Image Loading finish...")

	def next_sample_train(self):      
		self.cur_index_train += 1
		if self.cur_index_train >= self.n_train:
			self.cur_index_train = 0
			random.shuffle(self.indexes_train)
		return self.imgs_train[self.indexes_train[self.cur_index_train]], self.texts_train[self.indexes_train[self.cur_index_train]]
	
	def next_sample_val(self):      
		self.cur_index_val += 1
		if self.cur_index_val >= self.n_val:
			self.cur_index_val = 0
			random.shuffle(self.indexes_val)
		return self.imgs_val[self.indexes_val[self.cur_index_val]], self.texts_val[self.indexes_val[self.cur_index_val]]

	
	def next_batch_train(self):       
		while True:
			X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])     # (bs, 128, 64, 1)
			Y_data = np.ones([self.batch_size, self.max_text_len]) * self.blank_label
			input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)  # (bs, 1)
			label_length = np.zeros((self.batch_size, 1))           # (bs, 1)
			source_str = []
			for i in range(self.batch_size):
				img, text = self.next_sample_train()
				img = img.T
				img = np.expand_dims(img, -1)
				X_data[i] = img
				Y_data[i,0:len(text)] = text_to_labels(text)
				label_length[i] = len(text)
				source_str.append(text)

			inputs = {
				'the_input': X_data,  
				'the_labels': Y_data,  
				'input_length': input_length,  
				'label_length': label_length,
				'source_str': source_str  # used for visualization only
			}
			outputs = {'ctc': np.zeros([self.batch_size])}   
			yield (inputs, outputs)
	
	def next_batch_val(self):       
		while True:
			X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])     # (bs, 128, 64, 1)
			Y_data = np.ones([self.batch_size, self.max_text_len]) * self.blank_label
			input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)  # (bs, 1)
			label_length = np.zeros((self.batch_size, 1))           # (bs, 1)
			source_str = []
			for i in range(self.batch_size):
				img, text = self.next_sample_val()
				img = img.T
				img = np.expand_dims(img, -1)
				X_data[i] = img
				Y_data[i,0:len(text)] = text_to_labels(text)
				label_length[i] = len(text)
				source_str.append(text)

			inputs = {
				'the_input': X_data,  
				'the_labels': Y_data,  
				'input_length': input_length,  
				'label_length': label_length,
				'source_str': source_str  # used for visualization only
			}
			outputs = {'ctc': np.zeros([self.batch_size])}   
			yield (inputs, outputs)
			

class VizCallback(keras.callbacks.Callback):

	def __init__(self, run_name, test_func, text_img_gen, num_display_words=6):
		self.test_func = test_func
		self.output_dir = os.path.join(
			OUTPUT_DIR, run_name)
		self.text_img_gen = text_img_gen
		self.num_display_words = num_display_words
		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)

	def show_edit_distance(self, num):
		num_left = num
		mean_norm_ed = 0.0
		mean_ed = 0.0
		while num_left > 0:
			word_batch = next(self.text_img_gen)[0]
			num_proc = min(word_batch['the_input'].shape[0], num_left)
			decoded_res = decode_batch(self.test_func, word_batch['the_input'][0:num_proc])
			for j in range(num_proc):
				edit_dist = editdistance.eval(decoded_res[j],
											  word_batch['source_str'][j])
				mean_ed += float(edit_dist)
				mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
			num_left -= num_proc
		mean_norm_ed = mean_norm_ed / num
		mean_ed = mean_ed / num
		print('\nOut of %d samples:  Mean edit distance:'
			  '%.3f Mean normalized edit distance: %0.3f'
			  % (num, mean_ed, mean_norm_ed))

	def on_epoch_end(self, epoch, logs={}):
		self.model.save_weights(
			os.path.join(self.output_dir, 'weights%02d.h5' % (epoch)))
		self.show_edit_distance(256)
		word_batch = next(self.text_img_gen)[0]
		res = decode_batch(self.test_func,
						   word_batch['the_input'][0:self.num_display_words])
		if word_batch['the_input'][0].shape[0] <= 256:
			cols = 2
		else:
			cols = 1
		for i in range(self.num_display_words):
			pylab.subplot(self.num_display_words // cols, cols, i + 1)
			if K.image_data_format() == 'channels_first':
				the_input = word_batch['the_input'][i, 0, :, :]
			else:
				the_input = word_batch['the_input'][i, :, :, 0]
			
			the_input = the_input.T
			pylab.imshow(the_input, cmap='Greys_r')
			pylab.xlabel(
				'Truth = \'%s\'\nDecoded = \'%s\'' %
				(word_batch['source_str'][i], res[i]))
		fig = pylab.gcf()
		fig.set_size_inches(10, 13)
		pylab.savefig(os.path.join(self.output_dir, 'e%02d.png' % (epoch)))
		pylab.close()