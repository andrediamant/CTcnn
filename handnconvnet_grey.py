from sklearn.metrics import roc_auc_score,classification_report
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, Callback, ReduceLROnPlateau
import numpy as np
import sys

class Histories(Callback):
	
	def __init__(self, validation_generator = None, train_generator = None):
		super(Histories, self).__init__()
		self.validation_generator = validation_generator
		self.train_generator = train_generator

	def on_train_begin(self, logs={}):
		self.aucs = []
		self.trainingaucs = []
		self.losses = []

	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))
		valid_steps = np.ceil(self.validation_generator.samples/self.validation_generator.batch_size)
		true_classes = self.validation_generator.classes
		predictions = self.model.predict_generator(generator = self.validation_generator, steps = valid_steps,workers=1)
		roc_auc = roc_auc_score(y_true = true_classes, y_score = np.ravel(predictions))

		self.aucs.append(round(roc_auc,3))
		print('Validation AUCS')
		print(self.aucs)

		valid_steps = np.ceil(self.train_generator.samples/self.train_generator.batch_size)
		true_classes = self.train_generator.classes
		predictions = self.model.predict_generator(generator = self.train_generator, steps = valid_steps,workers=1)
		roc_auc = roc_auc_score(y_true = true_classes, y_score = np.ravel(predictions))

		self.trainingaucs.append(round(roc_auc,3))
		print('Training AUCS')
		print(self.trainingaucs)

		return

class MultiGPUCheckpointCallback(Callback):

	def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
				 save_best_only=False, save_weights_only=False,
				 mode='auto', period=1):
		super(MultiGPUCheckpointCallback, self).__init__()
		self.base_model = base_model
		self.monitor = monitor
		self.verbose = verbose
		self.filepath = filepath
		self.save_best_only = save_best_only
		self.save_weights_only = save_weights_only
		self.period = period
		self.epochs_since_last_save = 0

		if mode not in ['auto', 'min', 'max']:
			warnings.warn('ModelCheckpoint mode %s is unknown, '
						  'fallback to auto mode.' % (mode),
						  RuntimeWarning)
			mode = 'auto'

		if mode == 'min':
			self.monitor_op = np.less
			self.best = np.Inf
		elif mode == 'max':
			self.monitor_op = np.greater
			self.best = -np.Inf
		else:
			if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
				self.monitor_op = np.greater
				self.best = -np.Inf
			else:
				self.monitor_op = np.less
				self.best = np.Inf

	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {}
		self.epochs_since_last_save += 1
		if self.epochs_since_last_save >= self.period:
			self.epochs_since_last_save = 0
			filepath = self.filepath.format(epoch=epoch + 1, **logs)
			if self.save_best_only:
				current = logs.get(self.monitor)
				if current is None:
					warnings.warn('Can save best model only with %s available, '
								  'skipping.' % (self.monitor), RuntimeWarning)
				else:
					if self.monitor_op(current, self.best):
						if self.verbose > 0:
							print('Epoch %05d: %s improved from %0.5f to %0.5f,'
								  ' saving model to %s'
								  % (epoch + 1, self.monitor, self.best,
									 current, filepath))
						self.best = current
						if self.save_weights_only:
							self.base_model.save_weights(filepath, overwrite=True)
						else:
							self.base_model.save(filepath, overwrite=True)
					else:
						if self.verbose > 0:
							print('Epoch %05d: %s did not improve' %
								  (epoch + 1, self.monitor))
			else:
				if self.verbose > 0:
					print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
				if self.save_weights_only:
					self.base_model.save_weights(filepath, overwrite=True)
				else:
					self.base_model.save(filepath, overwrite=True)


class TensorBoardWrapper(TensorBoard):
    '''Sets the self.validation_data property for use with TensorBoard callback.''' #Enables use of validation_generator with Tensorboard histograms.

    def __init__(self, batch_gen, nb_steps, **kwargs):
        super().__init__(**kwargs)
        self.batch_gen = batch_gen # The generator.
        self.nb_steps = nb_steps     # Number of times to call next() on the generator.

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in range(self.nb_steps):
            ib, tb = next(self.batch_gen)
            if imgs is None and tags is None:
                imgs = np.zeros((self.nb_steps * ib.shape[0], *ib.shape[1:]), dtype=np.float32)
                tags = np.zeros((self.nb_steps * tb.shape[0], *tb.shape[1:]), dtype=np.uint8)
            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
        self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]
        return super().on_epoch_end(epoch, logs)



def main():

	import pandas as pd
	from time import time
	from keras import applications, optimizers
	from keras.models import Sequential, Model, load_model
	from keras.layers import Dense, Dropout, Activation, Flatten
	from keras.layers import Conv2D, MaxPooling2D,PReLU, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization
	from keras.optimizers import SGD
	from keras.utils import np_utils, multi_gpu_model
	from keras.utils.vis_utils import plot_model
	from keras.datasets import mnist
	from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
	from keras import backend as K
	import tensorflow as tf

	from numpy.random import seed
	from tensorflow import set_random_seed


	######## Forces the GPUs to not pre-hog all the memory.
	config = tf.ConfigProto() 
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	K.set_session(sess)

	######## Initialize directories.
	rootdir = "/ssd/andrediamant/"
	imagedir = rootdir + "images_institution/IMAGES_DM_CENTER/"
	outcometype = 'DM/'
	logtitle = "2ndrun_2019"
	size = 512
	print(logtitle)

	######## Build the model.

	model = Sequential() #Initializes the model. Sequential (allows linear stacking) as opposed to Functional (more complex, more power). 

	model.add(Conv2D(32, (5, 5), input_shape=(size, size, 1))) #Number of filters, size of filters, initialize input shape, ONLY needed in your first layer, afterwards it auto-computes.
	model.add(MaxPooling2D(pool_size=(4, 4),strides=4))
	model.add(PReLU()) 
	#model.add(BatchNormalization())

	model.add(Conv2D(64, (3, 3)))
	model.add(MaxPooling2D(pool_size=(4, 4),strides=4))
	model.add(PReLU())
	#model.add(BatchNormalization())


	model.add(Conv2D(128, (3, 3)))
	model.add(MaxPooling2D(pool_size=(4, 4),strides=4))
	model.add(PReLU())
	#model.add(BatchNormalization())

	model.add(Flatten()) 
	model.add(Dense(256))
	model.add(PReLU())
	model.add(Dense(128))
	model.add(PReLU())
	model.add(Dropout(0.50))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))


	######## If starting from previously computed weights, (very basic transfer learning or reinitializing model training).

	bestweights = rootdir + "Python/models/" + outcometype + logtitle + "_weights.best.hdf5"

	# try:
	# 	model.load_weights(bestweights)
	# 	print(bestweights + " loaded.")
	# except:
	# 	print("Weights not found.")
	# 	pass

	# model = load_model('/ssd/andrediamant/Python/models/DM/bestmodel.h5')
	# model.load_weights('/ssd/andrediamant/Python/models/DM/bestmodel.best.hdf5')

	# for layer in model.layers[:9]: #Example of how to freeze all layers up until layer 9, to only re-train the fully connected layers.
	# 	layer.trainable = False	
	# print("Layers frozen until " + model.layers[9].name)

	######################################

	gpu_model = multi_gpu_model(model, gpus = 2) #This is what allows the model to use both (or more GPUS). Note there is no sharing of memory, just linearly improves the computation time. 

	print(model.summary()) # Instantly generates full summary.

	#Initialize variables & print to log.
	lr = 0.01
	momentum = 0.5
	batch_size = 32
	print('lr = ' + str(lr))
	print('momentum = ' + str(momentum))
	print('batch size = ' + str(batch_size))

	#Compilation step. Needed to initialize the model, pick loss, optimizer, etc.. 
	gpu_model.compile(loss='binary_crossentropy',
				 #optimizer=optimizers.Adam(),
				  optimizer=optimizers.SGD(lr=lr,momentum=momentum),
				  metrics=['accuracy','mse'])
	

	#Generation of input data. Two seperate generators for validation and train (as validation does not use data-augmentation)
	train_datagen = ImageDataGenerator(rotation_range=20, horizontal_flip= True, vertical_flip = True, width_shift_range = 0.4, height_shift_range = 0.4)
	validation_datagen = ImageDataGenerator()

	train_generator = train_datagen.flow_from_directory(
			imagedir + 'TRAIN',  # this is the target directory
			target_size=(size, size),  
			batch_size=batch_size,
			class_mode='binary',
			color_mode = 'grayscale')  # since we use binary_crossentropy loss, we need binary labels

	train_generator_forauc = train_datagen.flow_from_directory(
			imagedir + 'TRAIN',  # this is the target directory
			target_size=(size, size),  
			batch_size=batch_size,
			class_mode='binary',
			color_mode = 'grayscale',
			shuffle = False)  # No need to shuffle as this is just to compute a metric after training. 

	validation_generator = validation_datagen.flow_from_directory(
			imagedir + 'TEST',
			target_size=(size, size),
			batch_size=106, 				#I've manually set this so the validation process uses two batches instead of one. (Irrelevant for numerical purposes, but can run into memory issues otherwise) 
			class_mode='binary',    
			color_mode = 'grayscale',
			shuffle = False)  # No need to shuffle as this is just to compute a metric after training. 

	classweights = {0 : 1, 1: 1} # If you wanted to perform any sort of imbalance adjustments, this is where it would be.

	#This creates a png of the artchitecture. I found it to be not very useful. Pretty ugly.
	plot_model(model,show_shapes = True, to_file = '/ssd/andrediamant/Python/models/' + outcometype + logtitle + '.png')

	#This initializes TensorBoard, the best way to vizualize progress of your training.
	tensorboard = TensorBoard(log_dir='/ssd/andrediamant/Python/tensorboardlogs/' + outcometype + logtitle, write_graph=False)

	#### WORK IN PROGRESS
	#tensorboard = TensorBoardWrapper(validation_generator,nb_steps=2,log_dir='/ssd/andrediamant/Python/tensorboardlogs/' + outcometype + logtitle, write_graph=False, histogram_freq=1,batch_size=batch_size,write_grads=True)
	####

	#Checkpoints save weights throughout. MultiGPU class is neccessary to fix a bug.
	checkpoints=MultiGPUCheckpointCallback(bestweights, model, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

	#Optional. After a certain amount of epochs without improvement, you can automatically reduce the learning rate.
	reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, verbose=1,min_lr=0.001)

	#Where the magic happens. Callbacks is where you add things to monitor the training. Histories in specific is a custom class to track AUCs. 
	history = gpu_model.fit_generator(
			train_generator,
			class_weight = classweights,
			steps_per_epoch= 4000 // batch_size,
			epochs=200,
			validation_data=validation_generator,
			validation_steps= 1,
			verbose = 2,
			callbacks = [tensorboard,Histories(validation_generator,train_generator_forauc),checkpoints,reducelr]
			)

	model.save('/ssd/andrediamant/Python/models/' + outcometype + logtitle + ".h5")  
	model.load_weights(bestweights)

	#This is all to get an idea of the type of scores the final model is generating and verify images are in the correct folder.
	valid_steps = np.ceil(train_generator_forauc.samples/train_generator_forauc.batch_size)
	probabilities = gpu_model.predict_generator(train_generator_forauc, valid_steps,verbose=1,workers=1)
	print(probabilities)
	print(train_generator_forauc.class_indices)
	print(train_generator_forauc.classes)

	valid_steps = np.ceil(validation_generator.samples/validation_generator.batch_size)
	probabilities = gpu_model.predict_generator(validation_generator, valid_steps,verbose=1,workers=1)
	print(probabilities)
	print(validation_generator.class_indices)
	print(validation_generator.classes)


if __name__ == "__main__":
	x = main()
