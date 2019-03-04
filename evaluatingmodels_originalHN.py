import matplotlib
matplotlib.use('Agg') #to actually be able to use matplotlib
import numpy as np 
from scipy.misc import imsave,imread
import time
from keras import backend as K
from vis.visualization import visualize_activation,visualize_saliency,visualize_cam,get_num_filters
from vis.utils import utils
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, confusion_matrix
from matplotlib import pyplot as plt
import itertools
import os
from keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import applications, optimizers, activations
from vis.input_modifiers import Jitter
from PIL import Image
import matplotlib.gridspec as gridspec


# util function to convert a tensor into a valid image
def deprocess_image(x):
	# normalize tensor: center on 0., ensure std is 0.1
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1

	# clip to [0, 1]
	x += 0.5
	x = np.clip(x, 0, 1)


	return x

def plot_confusion_matrix(cm, classes,
						  normalize=True,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	#print(cmnormalized)

	plt.figure()
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt), 
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig('confusionmatrix.eps')


def get_activations(model, model_inputs, print_shape_only=True, layer_name=None):
	print('----- activations -----')
	activations = []
	inp = model.input

	model_multi_inputs_cond = True
	if not isinstance(inp, list):
		# only one input! let's wrap it in a list.
		inp = [inp]
		model_multi_inputs_cond = False

	outputs = [layer.output for layer in model.layers if
			   layer.name == layer_name or layer_name is None]  # all layer outputs

	funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

	if model_multi_inputs_cond:
		list_inputs = []
		list_inputs.extend(model_inputs)
		list_inputs.append(0.)
	else:
		list_inputs = [model_inputs, 0.]

	# Learning phase. 0 = Test mode (no dropout or batch normalization)
	# layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
	layer_outputs = [func(list_inputs)[0] for func in funcs]
	for layer_activations in layer_outputs:
		activations.append(layer_activations)
		if print_shape_only:
			print(layer_activations.shape)
		else:
			print(layer_activations)
	return activations

def display_activations(activation_maps):
	import numpy as np

	batch_size = activation_maps[0].shape[0]
	assert batch_size == 1, 'One image at a time to visualize.'
	for i, activation_map in enumerate(activation_maps):
		print('Displaying activation map {}'.format(i))
		shape = activation_map.shape
		if len(shape) == 4:
			activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
		elif len(shape) == 2:
			# try to make it square as much as possible. we can skip some activations.
			activations = activation_map[0]
			num_activations = len(activations)
			if num_activations < 1024:  # too hard to display it on the screen.
				square_param = int(np.floor(np.sqrt(num_activations)))
				activations = activations[0: square_param * square_param]
				activations = np.reshape(activations, (square_param, square_param))
			else:
				activations = np.expand_dims(activations, axis=0)
		else:
			raise Exception('len(shape) = 3 has not been implemented.')
		plt.figure()
		plt.imshow(activations, interpolation='None', cmap='gray')
		#plt.show()	
		plt.savefig('activations_noDMcase_C15.png')


def generate_max_activation(model,gradmod,backprop):
	activations = visualize_activation(model,-1,filter_indices=[0],verbose=True,input_modifiers=[Jitter(16)],backprop_modifier=backprop,grad_modifier=gradmod,act_max_weight=1, lp_norm_weight=10,tv_weight=10)
	plt.imsave('activations_inferno.eps',activations[:,:,0],cmap='inferno')
	plt.imsave('activations_plasma.eps',activations[:,:,0],cmap='plasma')
	plt.imsave('activations_magma.eps',activations[:,:,0],cmap='magma')
	plt.imsave('activations_gray.eps',activations[:,:,0],cmap='gray')
	plt.imsave('activations_viridis.eps',activations[:,:,0],cmap='viridis')

def generate_saliency_cam_maps(model,testimage,camlayerofinterest,gradmod,backprop,imagename):
	grads = visualize_saliency(model,-1,filter_indices = None,seed_input= testimage,backprop_modifier = backprop,grad_modifier=gradmod)
	plt.imsave('saliency-' + imagename +'.png',grads,cmap='gray')
	cam=visualize_cam(model,camlayerofinterest,filter_indices = None,seed_input=testimage,backprop_modifier=backprop,grad_modifier=gradmod,penultimate_layer_idx=None) #0, 3, 6 are the conv layers. (so 1, 4 7 are what we want)
	plt.imsave('cam_conv3-' + imagename +'.png',cam)
	background = Image.open("currentCT.png")
	overlay = Image.open('cam_conv3-' + imagename +'.png')
	mergedimage=Image.blend(background,overlay,0.60)
	mergedimage.save("mergedimage.png","PNG")

def brute_force_montage(desiredslice,imagedir,model,camlayerofinterest,gradmod,backprop):
	padding = 30

	fig = plt.figure(figsize=(3,3))
	gs1 = gridspec.GridSpec(4, 3,wspace=0.05,hspace=0)
	imagefile = "HN-HMR-027-" + desiredslice
	ax = plt.subplot(gs1[0])
	ax.set_title("(a)",weight='bold',family='sans-serif')
	ax.set_ylabel("DM",size='large')
	plt.axis('off')
	testimage = imread(imagedir + "TEST/dm/" + imagefile + '.png')
	indcolumns = np.nonzero(testimage.any(axis=0))[0] # indices of non empty columns
	indrows = np.nonzero(testimage.any(axis=1))[0] # indices of non empty rows 
	plt.imshow(testimage[indrows[0]-padding-2:indrows[-1]+padding+3,indcolumns[0]-padding-35:indcolumns[-1]+padding+36],cmap = 'gray')
	plt.imsave('currentCT.png',testimage[indrows[0]-padding-2:indrows[-1]+padding+3,indcolumns[0]-padding-35:indcolumns[-1]+padding+36],cmap='gray')
	testimage = testimage.reshape(1,512,512,1)
	cam=visualize_cam(model,camlayerofinterest,filter_indices = None,seed_input=testimage,backprop_modifier=backprop,grad_modifier=gradmod,penultimate_layer_idx=None) #0, 3, 6 are the conv layers. (so 1, 4 7 are what we want)
	plt.imsave('cam_conv3-' + imagefile +'.png',cam[indrows[0]-padding-2:indrows[-1]+padding+3,indcolumns[0]-padding-35:indcolumns[-1]+padding+36])
	ax=plt.subplot(gs1[1])
	ax.set_title("(b)",weight='bold',family='sans-serif')
	plt.axis('off')
	plt.imshow(cam[indrows[0]-padding-2:indrows[-1]+padding+3,indcolumns[0]-padding-35:indcolumns[-1]+padding+36])
	background = Image.open("currentCT.png")
	overlay = Image.open('cam_conv3-' + imagefile +'.png')
	mergedimage=Image.blend(background,overlay,0.60)
	ax=plt.subplot(gs1[2])
	ax.set_title("(c)",weight='bold',family='sans-serif')
	plt.axis('off')
	plt.imshow(mergedimage)

	imagefile = "HN-HMR-011-" + desiredslice
	plt.subplot(gs1[3])
	plt.axis('off')
	testimage = imread(imagedir + "TEST/dm/" + imagefile + '.png')
	indcolumns = np.nonzero(testimage.any(axis=0))[0] # indices of non empty columns
	indrows = np.nonzero(testimage.any(axis=1))[0] # indices of non empty rows 	
	plt.imshow(testimage[indrows[0]-padding:indrows[-1]+padding,indcolumns[0]-padding:indcolumns[-1]+padding],cmap = 'gray')
	plt.imsave('currentCT.png',testimage[indrows[0]-padding:indrows[-1]+padding,indcolumns[0]-padding:indcolumns[-1]+padding],cmap='gray')
	testimage = testimage.reshape(1,512,512,1)
	cam=visualize_cam(model,camlayerofinterest,filter_indices = None,seed_input=testimage,backprop_modifier=backprop,grad_modifier=gradmod,penultimate_layer_idx=None) #0, 3, 6 are the conv layers. (so 1, 4 7 are what we want)
	plt.imsave('cam_conv3-' + imagefile +'.png',cam[indrows[0]-padding:indrows[-1]+padding,indcolumns[0]-padding:indcolumns[-1]+padding])
	plt.subplot(gs1[4])
	plt.axis('off')
	plt.imshow(cam[indrows[0]-padding:indrows[-1]+padding,indcolumns[0]-padding:indcolumns[-1]+padding])
	background = Image.open("currentCT.png")
	overlay = Image.open('cam_conv3-' + imagefile +'.png')
	mergedimage=Image.blend(background,overlay,0.60)
	plt.subplot(gs1[5])
	plt.axis('off')
	plt.imshow(mergedimage)

	imagefile = "HN-CHUM-015-" + desiredslice
	plt.subplot(gs1[6])
	plt.axis('off')
	testimage = imread(imagedir + "TEST/nodm/" + imagefile + '.png')
	indcolumns = np.nonzero(testimage.any(axis=0))[0] # indices of non empty columns
	indrows = np.nonzero(testimage.any(axis=1))[0] # indices of non empty rows 
	plt.imshow(testimage[indrows[0]-padding-9:indrows[-1]+padding+9,indcolumns[0]-padding-37:indcolumns[-1]+padding+37],cmap = 'gray')
	plt.imsave('currentCT.png',testimage[indrows[0]-padding-9:indrows[-1]+padding+9,indcolumns[0]-padding-37:indcolumns[-1]+padding+37],cmap='gray')
	testimage = testimage.reshape(1,512,512,1)
	cam=visualize_cam(model,camlayerofinterest,filter_indices = None,seed_input=testimage,backprop_modifier=backprop,grad_modifier=gradmod,penultimate_layer_idx=None) #0, 3, 6 are the conv layers. (so 1, 4 7 are what we want)
	plt.imsave('cam_conv3-' + imagefile +'.png',cam[indrows[0]-padding-9:indrows[-1]+padding+9,indcolumns[0]-padding-37:indcolumns[-1]+padding+37])
	plt.subplot(gs1[7])
	plt.axis('off')
	plt.imshow(cam[indrows[0]-padding-9:indrows[-1]+padding+9,indcolumns[0]-padding-37:indcolumns[-1]+padding+37])
	background = Image.open("currentCT.png")
	overlay = Image.open('cam_conv3-' + imagefile +'.png')
	mergedimage=Image.blend(background,overlay,0.60)
	plt.subplot(gs1[8])
	plt.axis('off')
	plt.imshow(mergedimage)


	imagefile = "HN-HMR-010-" + desiredslice
	plt.subplot(gs1[9])
	plt.axis('off')
	testimage = imread(imagedir + "TEST/nodm/" + imagefile + '.png')
	indcolumns = np.nonzero(testimage.any(axis=0))[0] # indices of non empty columns
	indrows = np.nonzero(testimage.any(axis=1))[0] # indices of non empty rows 
	plt.imshow(testimage[indrows[0]-padding-16:indrows[-1]+padding+17,indcolumns[0]-padding-45:indcolumns[-1]+padding+46],cmap = 'gray')
	plt.imsave('currentCT.png',testimage[indrows[0]-padding-16:indrows[-1]+padding+17,indcolumns[0]-padding-45:indcolumns[-1]+padding+46],cmap='gray')
	testimage = testimage.reshape(1,512,512,1)
	cam=visualize_cam(model,camlayerofinterest,filter_indices = None,seed_input=testimage,backprop_modifier=backprop,grad_modifier=gradmod,penultimate_layer_idx=None) #0, 3, 6 are the conv layers. (so 1, 4 7 are what we want)
	plt.imsave('cam_conv3-' + imagefile +'.png',cam[indrows[0]-padding-16:indrows[-1]+padding+17,indcolumns[0]-padding-45:indcolumns[-1]+padding+46])
	plt.subplot(gs1[10])
	plt.axis('off')
	plt.imshow(cam[indrows[0]-padding-16:indrows[-1]+padding+17,indcolumns[0]-padding-45:indcolumns[-1]+padding+46])
	background = Image.open("currentCT.png")
	overlay = Image.open('cam_conv3-' + imagefile +'.png')
	mergedimage=Image.blend(background,overlay,0.60)
	plt.subplot(gs1[11])
	plt.axis('off')
	plt.imshow(mergedimage)	
	plt.savefig("fullmontage.eps",dpi=300)

def thorough_numerical_evaluation(model,validation_generator,training_generator,threshold):
	probabilities = model.predict_generator(validation_generator, 1,verbose=1)
	score = model.evaluate_generator(validation_generator)
	
	true_classes = training_generator.classes
	predictions = model.predict_generator(generator = training_generator, steps = 2,workers=1)
	roc_auc = roc_auc_score(y_true = true_classes, y_score = np.ravel(predictions))
	print('Training AUC')
	print(roc_auc)

	true_classes = validation_generator.classes
	predictions = model.predict_generator(generator = validation_generator, steps = 2,workers=1)
	roc_auc = roc_auc_score(y_true = true_classes, y_score = np.ravel(predictions))
	print('Testing AUC')
	print(roc_auc)



	predictions = (predictions-min(predictions))/(max(predictions)-min(predictions)) #Normalize between 0 and 1.
	print(predictions)
	print(validation_generator.filenames)

	fpr, tpr, thresholds = roc_curve(true_classes, np.ravel(predictions))
	print('Specificity (1-FPR)')
	specificity = 1- fpr
	print(specificity)
	print('TPR (sensitivity)')
	print(tpr)
	print('Thresholds')
	print(thresholds)
	plt.figure()
	plt.tight_layout()
	plt.plot(fpr,tpr,color = 'darkorange',label = 'ROC curve (area = %0.2f)'% roc_auc)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])	
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic curve')
	plt.legend(loc="lower right")

	ax2 = plt.gca().twinx()
	ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')
	ax2.set_ylabel('Threshold',color='r')
	ax2.set_ylim([thresholds[-1],thresholds[0]])
	ax2.set_xlim([fpr[0],fpr[-1]])

	plt.savefig('ROCcurve.eps')

	dmindex = predictions <= threshold
	nodmindex = predictions > threshold
	predictions[dmindex] = 0
	predictions[nodmindex] = 1
	print(predictions)
	cnf = confusion_matrix(true_classes,predictions)
	plot_confusion_matrix(cnf,['dm','nodm'],normalize=False)

	print(validation_generator.class_indices)
	print(validation_generator.classes)
	print(model.metrics_names)
	print(score)

def get_weights(layer_name,model):
	layer_idx = utils.find_layer_idx(model, layer_name)
	print(np.shape(model.layers[layer_idx].get_weights()[0]))
	weights= model.layers[layer_idx].get_weights()[0][:,:,0,:] #For CONV layers, NOTE THIS IS ONLY LOOKING AT A SINGLE SLICE OF THE FILTERS WEIGHTS, just to show it's possible, I have never explicitly used this.
	#3weights= model.layers[layer_idx].get_weights()[0] #FOR PRELU layers
	plt.figure()
	print(np.shape(weights))
	for i in range(1,np.shape(weights)[2]):
		plt.subplot(12,12,i)
		plt.imshow(weights[:,:,i],interpolation="nearest",cmap="gray")
		plt.axis('off')
	plt.savefig('weights.png')

def generate_filter_max_activation(model,layer_name):
	input_img = model.input
	img_width = 512
	img_height = 512

	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	kept_filters = []

	def normalize(x):
		# utility function to normalize a tensor by its L2 norm
		return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

	for filter_index in range(128): #Set this the number of filters in the layer you are interested in.
		print('Processing filter %d' % filter_index)
		start_time = time.time()

		# we build a loss function that maximizes the activation
		# of the nth filter of the layer considered
		layer_output = layer_dict[layer_name].output
		if K.image_data_format() == 'channels_first':
			loss = K.mean(layer_output[:, filter_index, :, :])
		else:
			loss = K.mean(layer_output[:, :, :, filter_index])

		# we compute the gradient of the input picture wrt this loss
		grads = K.gradients(loss, input_img)[0]

		# normalization trick: we normalize the gradient
		grads = normalize(grads)

		# this function returns the loss and grads given the input picture
		iterate = K.function([input_img], [loss, grads])

		# step size for gradient ascent
		step = 1 

		# we start from a gray image with some random noise
		if K.image_data_format() == 'channels_first':
			input_img_data = np.random.random((1, 1, img_width, img_height))
		else:
			input_img_data = np.random.random((1, img_width, img_height, 1))
		input_img_data = (input_img_data - 0.5) * 20 

		# we run gradient ascent for x steps
		for i in range(400):
			loss_value, grads_value = iterate([input_img_data])
			input_img_data += grads_value * step

			if i == 399:
				print('Current loss value:', loss_value)

			if loss_value <= 0.:
			## some filters get stuck to 0, we can skip them
				print('break')
				break

		# decode the resulting input image
		if loss_value > 0:
			img = deprocess_image(input_img_data[0])
			kept_filters.append((img, loss_value))

		end_time = time.time()
		print('Filter %d processed in %ds' % (filter_index, end_time - start_time))
		print(np.shape(kept_filters))

def brute_force_filter_stich(kept_filters):
	# we will stich the best 64 filters on a 8 x 8 grid.
	n = 2

	kept_filters.sort(key=lambda x: x[1], reverse=True)
	#kept_filters = kept_filters[:n * n]
	print(np.shape(kept_filters))

	# build a black picture with enough space for
	# our 8 x 8 filters of size 128 x 128, with a 5px margin in between

	margin = 5	
	#width = (n+1) * img_width + (n) * margin #HEIGHT MODIFIED
	width = n * img_width + (n-1) * margin 
	height = n * img_height + (n-1) * margin 
	stitched_filters = np.zeros((width, height, 1))

	#fill the picture with our saved filters
	# for i in range(n):
	# 	for j in range(n):
	# 		print(i*n+j)
	# 		img, loss = kept_filters[i * n + j]
	# 		stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
	# 						 (img_height + margin) * j: (img_height + margin) * j + img_height,:] = img
	# 		plt.imsave('filter_%03d.png' % (i*n+j),img[:,:,0],cmap='gist_ncar')

	# for i in range(7):
	# 	print(i+121)
	# 	img, loss = kept_filters[i+121]
	# 	stitched_filters[(img_width + margin) * 11: (img_width + margin) * 11 + img_width,
	# 					 (img_height + margin) * i: (img_height + margin) * i + img_height,:] = img
	# 	plt.imsave('filter_%03d.png' % (i+121),img[:,:,0],cmap='gray')		

	img, loss = kept_filters[1]
	stitched_filters[(img_width + margin) * 0: (img_width + margin) * 0 + img_width,
					 (img_height + margin) * 0: (img_height + margin) * 0 + img_height,:] = img

	img, loss = kept_filters[4]
	stitched_filters[(img_width + margin) * 0: (img_width + margin) * 0 + img_width,
					 (img_height + margin) * 1: (img_height + margin) * 1 + img_height,:] = img

	img, loss = kept_filters[11]
	stitched_filters[(img_width + margin) * 1: (img_width + margin) * 1 + img_width,
					 (img_height + margin) * 0: (img_height + margin) * 0 + img_height,:] = img

	img, loss = kept_filters[16]
	stitched_filters[(img_width + margin) * 1: (img_width + margin) * 1 + img_width,
					 (img_height + margin) * 1: (img_height + margin) * 1 + img_height,:] = img					 					 

	stitched_filters=stitched_filters[:,:,0]
	# save the result to disk
	plt.imsave('stitched_filters_%dx%d_gist.eps' % (n, n), stitched_filters,cmap='gist_ncar')
	#plt.imsave('stitched_filters_%dx%d_gray.eps' % (n, n), stitched_filters,cmap='gray')
	#plt.imsave('stitched_filters_%dx%d_jet.eps' % (n, n), stitched_filters,cmap='jet')

def main():
	rootdir = "/ssd/andrediamant/images_institution/"
	desiredslice = 'center'
	outcometype = 'DM'
	imagedir = rootdir  + "IMAGES_" + outcometype + "_BOTTOM_CROPPED/"
	modelname = "/transferlearning_freshcap" 
	imagefile = "HN-HMR-011-bottom"
	
	#testimage = imread(imagedir + "TEST/dm/" + imagefile + '.png')
	#plt.imsave('currentCT.png',testimage,cmap='gray')
	#testimage = testimage.reshape(1,512,512,1)


	#Initialize parameters.
	gradmod = 'relu'
	backprop = 'guided'
	camlayerofinterest = 7
	threshold = 0.7


	#Create the generator for the set of images we're interested in evaluating.
	generator = ImageDataGenerator()
	validation_generator = generator.flow_from_directory(
			imagedir + 'TEST',
			target_size=(224, 224),
			batch_size=53,
			class_mode='binary',
			shuffle = False,
			color_mode='rgb')

	training_generator = generator.flow_from_directory(
			imagedir + 'TRAIN',
			target_size=(224, 224),
			batch_size=96,
			class_mode='binary',
			shuffle = False,
			color_mode='rgb')

#	np.savetxt('testfilenames.txt',validation_generator.filenames,delimiter=',',fmt='%s')
#	np.savetxt('trainfilenames.txt',training_generator.filenames,delimiter=',',fmt='%s')

	#Load the model and (optional) its best weights.
	model = load_model('/ssd/andrediamant/Python/models/' + outcometype + modelname  + '.h5')
	#model.load_weights('/ssd/andrediamant/Python/models/' + outcometype + modelname  +'_weights.best.hdf5')
	model.compile(loss='binary_crossentropy',
				  optimizer=optimizers.SGD(lr=0.01,momentum=0.5),
				  metrics=['accuracy','mse'])
	print(model.summary())

	predictions = model.predict_generator(generator = validation_generator, steps = 2, workers = 1)
#	np.savetxt('testoutputs.csv',predictions,delimiter=',',fmt='%1.3f')

	predictions = model.predict_generator(generator = training_generator, steps = 2, workers = 1)
#	np.savetxt('trainoutputs.csv',predictions,delimiter=',',fmt='%1.3f')

	layer_name = 'dense_1'
	intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
	print(intermediate_layer_model.summary())
	intermediate_output = intermediate_layer_model.predict_generator(generator = validation_generator, steps = 2,workers=1)
	print(np.shape(intermediate_output))

#	np.savetxt('testdense1output.csv',intermediate_output,delimiter=',',fmt='%1.3f')
#	np.savetxt('testoutcomes.csv',validation_generator.classes,delimiter=',',fmt='%1.3f')
	
	intermediate_output = intermediate_layer_model.predict_generator(generator = training_generator, steps = 2,workers=1)

#	np.savetxt('traindense1output.csv',intermediate_output,delimiter=',',fmt='%1.3f')
#	np.savetxt('trainoutcomes.csv',training_generator.classes,delimiter=',',fmt='%1.3f')


	#Example of how to get the weights from the second last layer.
	#print(model.layers[-2].get_weights())

	# generate_max_activation(model,gradmod,backprop)

	# generate_saliency_cam_maps(model,testimage,camlayerofinterest,gradmod,backprop,'test')

	# #brute_force_montage(desiredslice,imagedir,model,camlayerofinterest,gradmod,backprop)

	thorough_numerical_evaluation(model,validation_generator,training_generator,threshold)

	# os.chdir('/ssd/andrediamant/Python/workingimagedir') 
	# layer_name = 'dense_1'

	# get_weights(layer_name,model)

	# activations=get_activations(model, testimage, print_shape_only=True, layer_name=layer_name)
	# display_activations(activations)

	# generate_filter_max_activation(model,layer_name)

	#brute_force_filter_stich(kept_filters)


if __name__ == "__main__":
	x = main()
