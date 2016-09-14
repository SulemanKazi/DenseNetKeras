from keras.models import Model
from keras.layers import Input, Activation, merge
from keras.layers import Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Dense
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization

def addLayer(previousLayer, nChannels, nOutChannels, dropRate, blockNum):

	bn = BatchNormalization(name = 'denseb_BatchNorm_{}'.format(blockNum) , axis = 1)(previousLayer)

	relu = Activation('relu', name ='denseb_relu_{}'.format(blockNum))(bn)

	conv = Convolution2D(nOutChannels, 3, 3, border_mode='same', name='denseb_conv_{}'.format(blockNum))(relu)

	if dropRate is not None:

		dp = Dropout(dropRate, name='denseb_dropout_{}'.format)(conv)

		return merge([dp, previousLayer], mode='concat', concat_axis=1)

	else:

		return merge([conv, previousLayer], mode='concat', concat_axis=1)


def addTransition(previousLayer, nChannels, nOutChannels, dropRate, blockNum):

	bn = BatchNormalization(name = 'tr_BatchNorm_{}'.format(blockNum), axis = 1)(previousLayer)

	relu = Activation('relu', name ='tr_relu_{}'.format(blockNum))(bn)

	conv = Convolution2D(nOutChannels, 1, 1, border_mode='same', name='tr_conv_{}'.format(blockNum))(relu)

	if dropRate is not None:

		dp = Dropout(dropRate, name='tr_dropout_{}'.format)(conv)

		avgPool = AveragePooling2D(pool_size=(2, 2))(dp)

	else:
		avgPool = AveragePooling2D(pool_size=(2, 2))(conv)

	return avgPool

def createModel(depth, inputShape =(3, 32, 32), dataset ='cifar10'):

	if (depth - 4) % 3 != 0:
		raise Exception('Depth must be 3n + 4!')

	#Layers in each denseblock
	N = (depth - 4) / 3

	#Growth rate
	growthRate = 12

	#DropOut Rate
	dropRate = None

	# Channels before entering the first denseblock
    # set it to be comparable with growth rate
	nChannels = 16

	numBlocksAdded = 0

	input_img = Input(shape = inputShape )

	previousLayer = Convolution2D(nChannels, 3, 3, border_mode='same', name='conv1')(input_img)


	for i in range(N):
		previousLayer = addLayer(previousLayer, nChannels, growthRate, dropRate, i + numBlocksAdded * N)
		nChannels += growthRate

	previousLayer = addTransition(previousLayer, nChannels, nChannels, dropRate, 1)

	numBlocksAdded += 1

	for i in range(N):
		previousLayer = addLayer(previousLayer, nChannels, growthRate, dropRate, i + numBlocksAdded * N)
		nChannels += growthRate

	previousLayer = addTransition(previousLayer, nChannels, nChannels, dropRate, 2)

	numBlocksAdded += 1

	for i in range(N):
		previousLayer = addLayer(previousLayer, nChannels, growthRate, dropRate, i + numBlocksAdded * N)
		nChannels += growthRate	

	numBlocksAdded += 1


	bn = BatchNormalization(name = 'BatchNorm_final', axis = 1)(previousLayer)

	relu = Activation('relu', name ='relu_final')(bn)

	avgPool = AveragePooling2D(pool_size=(8, 8), name = 'avg_pool_final')(relu)

	# Flatten
	flatten = Flatten(name='flatten')(avgPool)

	if dataset == 'cifar10':
		model_out = Dense(10, activation = 'softmax')(flatten)

	elif dataset == 'cifar100':
		model_out = Dense(100, activation = 'softmax')(flatten)

	return Model(input=input_img, output=model_out)


