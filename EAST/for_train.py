from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, Callback
import tensorflow as tf
import keras.backend as K
from PIL import Image
import io


class CustomModelCheckpoint(Callback):
	def __init__(self, model, path, period, save_weights_only):
		super(CustomModelCheckpoint, self).__init__()
		self.period = period
		self.path = path
		# We set the model (non multi gpu) under an other name
		self.model_for_saving = model
		self.epochs_since_last_save = 0
		self.save_weights_only = save_weights_only

	def on_epoch_end(self, epoch, logs=None):
		self.epochs_since_last_save += 1
		if self.epochs_since_last_save >= self.period:
			self.epochs_since_last_save = 0
			if self.save_weights_only:
				self.model_for_saving.save_weights(self.path.format(epoch=epoch + 1, **logs), overwrite=True)
			else:
				self.model_for_saving.save(self.path.format(epoch=epoch + 1, **logs), overwrite=True)

def make_image_summary(tensor):
	"""
	Convert an numpy representation image to Image protobuf.
	Copied from https://github.com/lanpa/tensorboard-pytorch/
	"""    
	if len(tensor.shape) == 2:
		height, width = tensor.shape
		channel = 1
	else:
		height, width, channel = tensor.shape
		if channel == 1:
			tensor = tensor[:, :, 0]
	image = Image.fromarray(tensor)
	output = io.BytesIO()
	image.save(output, format='PNG')
	image_string = output.getvalue()
	output.close()
	return tf.Summary.Image(height=height,
						 width=width,
						 colorspace=channel,
						 encoded_image_string=image_string)
				
class CustomTensorBoard(TensorBoard):
	def __init__(self, log_dir, score_map_loss_weight, small_text_weight, data_generator, FLAGS, write_graph=False):
		self.score_map_loss_weight = score_map_loss_weight
		self.small_text_weight = small_text_weight
		self.data_generator = data_generator
		self.FLAGS =FLAGS
		super(CustomTensorBoard, self).__init__(log_dir=log_dir, write_graph=write_graph)

	def on_epoch_end(self, epoch, logs=None):        
		logs.update({'learning_rate': K.eval(self.model.optimizer.lr), 'small_text_weight': K.eval(self.small_text_weight)})
		data = next(self.data_generator)
		pred_score_maps, pred_geo_maps = self.model.predict([data[0][0], data[0][1], data[0][2], data[0][3]])
		img_summaries = []
		for i in range(3):
			input_image_summary = make_image_summary(((data[0][0][i] + 1) * 127.5).astype('uint8'))
			overly_small_text_region_training_mask_summary = make_image_summary((data[0][1][i] * 255).astype('uint8'))
			text_region_boundary_training_mask_summary = make_image_summary((data[0][2][i] * 255).astype('uint8'))
			target_score_map_summary = make_image_summary((data[1][0][i] * 255).astype('uint8'))
			pred_score_map_summary = make_image_summary((pred_score_maps[i] * 255).astype('uint8'))            
			img_summaries.append(tf.Summary.Value(tag='input_image/%d' % i, image=input_image_summary))
			img_summaries.append(tf.Summary.Value(tag='overly_small_text_region_training_mask/%d' % i, image=overly_small_text_region_training_mask_summary))
			img_summaries.append(tf.Summary.Value(tag='text_region_boundary_training_mask/%d' % i, image=text_region_boundary_training_mask_summary))
			img_summaries.append(tf.Summary.Value(tag='score_map_target/%d' % i, image=target_score_map_summary))
			img_summaries.append(tf.Summary.Value(tag='score_map_pred/%d' % i, image=pred_score_map_summary))
			for j in range(4):
				target_geo_map_summary = make_image_summary((data[1][1][i, :, :, j] / self.FLAGS.input_size * 255).astype('uint8'))
				pred_geo_map_summary = make_image_summary((pred_geo_maps[i, :, :, j] / self.FLAGS.input_size * 255).astype('uint8'))
				img_summaries.append(tf.Summary.Value(tag='geo_map_%d_target/%d' % (j, i), image=target_geo_map_summary))
				img_summaries.append(tf.Summary.Value(tag='geo_map_%d_pred/%d' % (j, i), image=pred_geo_map_summary))
			target_geo_map_summary = make_image_summary(((data[1][1][i, :, :, 4] + 1) * 127.5).astype('uint8'))
			pred_geo_map_summary = make_image_summary(((pred_geo_maps[i, :, :, 4] + 1) * 127.5).astype('uint8'))
			img_summaries.append(tf.Summary.Value(tag='geo_map_%d_target/%d' % (4, i), image=target_geo_map_summary))
			img_summaries.append(tf.Summary.Value(tag='geo_map_%d_pred/%d' % (4, i), image=pred_geo_map_summary))
		tf_summary = tf.Summary(value=img_summaries)
		self.writer.add_summary(tf_summary, epoch + 1)
		super(CustomTensorBoard, self).on_epoch_end(epoch + 1, logs)
		
		
class SmallTextWeight(Callback):
	def __init__(self, weight):
		self.weight = weight

	# TO BE CHANGED
	def on_epoch_end(self, epoch, logs={}):
		#K.set_value(self.weight, np.minimum(epoch / (0.5 * FLAGS.max_epochs), 1.))
		K.set_value(self.weight, 0)

		
class ValidationEvaluator(Callback):
	def __init__(self, validation_data, validation_log_dir, FLAGS, period=5):
		super(Callback, self).__init__()

		self.period = period
		self.validation_data = validation_data
		self.validation_log_dir = validation_log_dir
		self.FLAGS = FLAGS
		self.val_writer = tf.summary.FileWriter(self.validation_log_dir)

	def on_epoch_end(self, epoch, logs={}):
		if (epoch + 1) % self.period == 0:
			val_loss, val_score_map_loss, val_geo_map_loss = self.model.evaluate([self.validation_data[0], self.validation_data[1], self.validation_data[2], self.validation_data[3]], [self.validation_data[3], self.validation_data[4]], batch_size=self.FLAGS.batch_size)
			print('\nEpoch %d: val_loss: %.4f, val_score_map_loss: %.4f, val_geo_map_loss: %.4f' % (epoch + 1, val_loss, val_score_map_loss, val_geo_map_loss))
			val_loss_summary = tf.Summary()
			val_loss_summary_value = val_loss_summary.value.add()
			val_loss_summary_value.simple_value = val_loss
			val_loss_summary_value.tag = 'loss'
			self.val_writer.add_summary(val_loss_summary, epoch + 1)
			val_score_map_loss_summary = tf.Summary()
			val_score_map_loss_summary_value = val_score_map_loss_summary.value.add()
			val_score_map_loss_summary_value.simple_value = val_score_map_loss
			val_score_map_loss_summary_value.tag = 'pred_score_map_loss'
			self.val_writer.add_summary(val_score_map_loss_summary, epoch + 1)
			val_geo_map_loss_summary = tf.Summary()
			val_geo_map_loss_summary_value = val_geo_map_loss_summary.value.add()
			val_geo_map_loss_summary_value.simple_value = val_geo_map_loss
			val_geo_map_loss_summary_value.tag = 'pred_geo_map_loss'
			self.val_writer.add_summary(val_geo_map_loss_summary, epoch + 1)

			pred_score_maps, pred_geo_maps = self.model.predict([self.validation_data[0][0:3], self.validation_data[1][0:3], self.validation_data[2][0:3], self.validation_data[3][0:3]])
			img_summaries = []
			for i in range(3):
				input_image_summary = make_image_summary(((self.validation_data[0][i] + 1) * 127.5).astype('uint8'))
				overly_small_text_region_training_mask_summary = make_image_summary((self.validation_data[1][i] * 255).astype('uint8'))
				text_region_boundary_training_mask_summary = make_image_summary((self.validation_data[2][i] * 255).astype('uint8'))
				target_score_map_summary = make_image_summary((self.validation_data[3][i] * 255).astype('uint8'))
				pred_score_map_summary = make_image_summary((pred_score_maps[i] * 255).astype('uint8'))            
				img_summaries.append(tf.Summary.Value(tag='input_image/%d' % i, image=input_image_summary))
				img_summaries.append(tf.Summary.Value(tag='overly_small_text_region_training_mask/%d' % i, image=overly_small_text_region_training_mask_summary))
				img_summaries.append(tf.Summary.Value(tag='text_region_boundary_training_mask/%d' % i, image=text_region_boundary_training_mask_summary))
				img_summaries.append(tf.Summary.Value(tag='score_map_target/%d' % i, image=target_score_map_summary))
				img_summaries.append(tf.Summary.Value(tag='score_map_pred/%d' % i, image=pred_score_map_summary))
				for j in range(4):
					target_geo_map_summary = make_image_summary((self.validation_data[4][i, :, :, j] / self.FLAGS.input_size * 255).astype('uint8'))
					pred_geo_map_summary = make_image_summary((pred_geo_maps[i, :, :, j] / self.FLAGS.input_size * 255).astype('uint8'))
					img_summaries.append(tf.Summary.Value(tag='geo_map_%d_target/%d' % (j, i), image=target_geo_map_summary))
					img_summaries.append(tf.Summary.Value(tag='geo_map_%d_pred/%d' % (j, i), image=pred_geo_map_summary))
				target_geo_map_summary = make_image_summary(((self.validation_data[4][i, :, :, 4] + 1) * 127.5).astype('uint8'))
				pred_geo_map_summary = make_image_summary(((pred_geo_maps[i, :, :, 4] + 1) * 127.5).astype('uint8'))
				img_summaries.append(tf.Summary.Value(tag='geo_map_%d_target/%d' % (4, i), image=target_geo_map_summary))
				img_summaries.append(tf.Summary.Value(tag='geo_map_%d_pred/%d' % (4, i), image=pred_geo_map_summary))
			tf_summary = tf.Summary(value=img_summaries)
			self.val_writer.add_summary(tf_summary, epoch + 1)
			self.val_writer.flush()