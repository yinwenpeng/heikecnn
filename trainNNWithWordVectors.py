#!/usr/bin/python

from myConvNN import *
from logistic_sgd import LogisticRegression, load_data
from mlp2 import HiddenLayer

import sys
import os
import time
#import pprint
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import mmap
import cPickle

if len(sys.argv) != 5:
  print "please pass the context training and validation files and the file for saving the net and the word vector file as parameters"
  exit(0)

trainfile = sys.argv[1]
devfile = sys.argv[2]
#evalfile = sys.argv[3]

# reading word vectors
wordvectors = {}
vectorsize = 0
f = open(sys.argv[4], 'r')
count = 0
for line in f:
  if count == 0:
    count += 1
    continue
  parts = line.split()
  word = parts[0]
  parts.pop(0)
  wordvectors[word] = parts
  vectorsize = len(parts)
f.close()

# input layer as concatenation of <before context> <name/filler> <between_skip context> <name/filler> <after context>

contextsize = 18 # 5 for contextBefore, 6 for contextBetween, 5 for contextAfter, 2 for <name> and <filler>
## get vocab
#vocabHash = {}
#f = open(trainfile, 'r')
#numSamples = 0
#for line in f:
#  line = line.strip()
#  parts = line.split(' : ')
#  slot = parts[0]
#  wholeContext = parts[1]
#  contextWords = wholeContext.split()
#  for w in contextWords:
#    vocabHash[w] = 1
#  numSamples += 1
#f.close()

f = open(trainfile, 'r+')
numSamples = 0
buf = mmap.mmap(f.fileno(), 0)
readline = buf.readline
while readline():
  numSamples += 1
f.close()

#ind = 0
#for w in vocabHash:
#  vocabHash[w] = ind
#  ind += 1
#vocabsize = len(wordvectors)

# get word vectors
#initialVector = []
inputMatrixTrain = numpy.empty(shape = (numSamples, vectorsize * contextsize))
resultVectorTrain = []
#for i in range(0, len(vocabHash)):
#  initialVector.append(0)
f = open(trainfile, 'r')
sample = 0
for line in f:
  line = line.strip()
  parts = line.split(' : ')
  posNeg = parts[0]
  if "+" in posNeg:
    resultVectorTrain.append(1)
  else:
    resultVectorTrain.append(0)
  wholeContext = parts[2]
  contextWords = wholeContext.split()
  matrix = numpy.zeros(shape = (vectorsize, contextsize))
  for i in range(0, len(contextWords)):
    word = contextWords[i]
    if word in wordvectors and i < contextsize:
      curVector = wordvectors[word]
      for j in range(0, vectorsize):
        if j > len(curVector):
          print "ERROR: mismatch in word vector lengths: " + str(len(curVector)) + " vs " + vectorsize
          exit()
        elem = curVector[j]
        matrix[j, i] = elem
    else:
      print "ERROR: reading unknown word " + word + " at position " + str(i)
  matrix = numpy.reshape(matrix, vectorsize * contextsize)
  inputMatrixTrain[sample,:] = matrix
  sample += 1
f.close()

# read dev file
numSamplesDev = 0
f = open(devfile, "r+")
buf = mmap.mmap(f.fileno(), 0) # seems to be faster than normal file read for line count
readline = buf.readline
while readline():
  numSamplesDev += 1
f.close()

f = open(devfile, 'r')
inputMatrixDev = numpy.empty(shape = (numSamplesDev, vectorsize * contextsize))
resultVectorDev = []
resultVectorEval = [] # eval calculation in another file!
inputMatrixEval = numpy.empty(shape = (numSamplesDev, vectorsize * contextsize)) # eval calculation in another file!
sample = 0
for line in f:
  line = line.strip()
  parts = line.split(' : ')
  posNeg = parts[0]
  if "+" in posNeg:
    resultVectorDev.append(1)
    resultVectorEval.append(1)
  else:
    resultVectorDev.append(0)
    resultVectorEval.append(0)
  wholeContext = parts[2]
  contextWords = wholeContext.split()
  matrix = numpy.zeros(shape = (vectorsize, contextsize))
  for i in range(0, len(contextWords)):
    word = contextWords[i]
    if word in wordvectors and i < contextsize:
      curVector = wordvectors[word]
      for j in range(0, vectorsize):
        if j > len(curVector):
          print "ERROR: mismatch in word vector lengths: " + str(len(curVector)) + " vs " + vectorsize
          exit()
        elem = curVector[j]
        matrix[j, i] = elem
  matrix = numpy.reshape(matrix, vectorsize * contextsize)
  inputMatrixDev[sample,:] = matrix
  inputMatrixEval[sample,:] = matrix
  sample += 1
f.close()

# train network
rng = numpy.random.RandomState(23455)

dt = theano.config.floatX
train_set_x = theano.shared(numpy.matrix(inputMatrixTrain, dtype = dt))
test_set_x = theano.shared(numpy.matrix(inputMatrixEval, dtype = dt))
valid_set_x = theano.shared(numpy.matrix(inputMatrixDev, dtype = dt))
train_set_y = theano.shared(numpy.array(resultVectorTrain, dtype = numpy.dtype(numpy.int32)))
test_set_y = theano.shared(numpy.array(resultVectorEval, dtype = numpy.dtype(numpy.int32)))
valid_set_y = theano.shared(numpy.array(resultVectorDev, dtype = numpy.dtype(numpy.int32)))

learning_rate=0.1
n_epochs=200
nkerns=[20, 50]
batch_size=2

# compute number of minibatches for training, validation and testing
n_train_batches = train_set_x.get_value(borrow=True).shape[0]
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
n_test_batches = test_set_x.get_value(borrow=True).shape[0]
n_train_batches /= batch_size
n_valid_batches /= batch_size
n_test_batches /= batch_size

# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')   # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

# TODO: set nkerns and filtersize properly, see below: poolsize ok?
ishape = [vectorsize, contextsize]  # this is the size of context matrizes
#print "vocabsize: " + str(vocabsize)
filtersize = [1,3]
pool = [1, ishape[1] - filtersize[1] + 1]

######################
# BUILD ACTUAL MODEL #
######################
print '... building the model'

# Reshape matrix of rasterized images of shape (batch_size,28*28)
# to a 4D tensor, compatible with our LeNetConvPoolLayer
layer0_input = x.reshape((batch_size, 1, ishape[0], ishape[1]))

# Construct the first convolutional pooling layer:
## filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
## maxpooling reduces this further to (24/2,24/2) = (12,12)
## 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
# filtering reduces the image size to (vocabsize-1+1,18-3+1) = (vocabsize, 16)
# maxpooling reduces this further to (vocabsize/1, 16/16) = (vocabsize, 1)
layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, ishape[0], ishape[1]),
            filter_shape=(nkerns[0], 1, filtersize[0], filtersize[1]), poolsize=(pool[0], pool[1]))

# Construct the second convolutional pooling layer
# filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
# maxpooling reduces this further to (8/2,8/2) = (4,4)
# 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
#layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
#            image_shape=(batch_size, nkerns[0], 12, 12),
#            filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))

## the HiddenLayer being fully-connected, it operates on 2D matrices of
## shape (batch_size,num_pixels) (i.e matrix of rasterized images).
## This will generate a matrix of shape (20,32*4*4) = (20,512)
#layer2_input = layer1.output.flatten(2)
layer2_input = layer0.output.flatten(2)

# construct a fully-connected sigmoidal layer
layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[0] * vectorsize * 1,
                         n_out=500, activation=T.tanh)

# classify the values of the fully-connected sigmoidal layer
layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=1)

# the cost we minimize during training is the NLL of the model
cost = layer3.negative_log_likelihood(y)

# create a function to compute the mistakes that are made by the model
test_model = theano.function([index], layer3.errors(y), 
         givens={ 
             x: test_set_x[index * batch_size: (index + 1) * batch_size], 
             y: test_set_y[index * batch_size: (index + 1) * batch_size]})

validate_model = theano.function([index], layer3.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

# create a list of all model parameters to be fit by gradient descent
#params = layer3.params + layer2.params + layer1.params + layer0.params
params = layer3.params + layer2.params + layer0.params

# create a list of gradients for all model parameters
grads = T.grad(cost, params)

# train_model is a function that updates the model parameters by
# SGD Since this model has many parameters, it would be tedious to
# manually create an update rule for each model parameter. We thus
# create the updates list by automatically looping over all
# (params[i],grads[i]) pairs.
updates = []
for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

#bla = train_set_x[index * batch_size: (index + 1) * batch_size]
#print bla.shape

train_model = theano.function([index], cost, updates=updates,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})

###############
# TRAIN MODEL #
###############
print '... training'
# early-stopping parameters
patience = 10000  # look as this many examples regardless
patience_increase = 2  # wait this much longer when a new best is
                           # found
improvement_threshold = 0.995  # a relative improvement of this much is
                                 # considered significant
validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

best_params = []
best_validation_loss = numpy.inf
best_iter = 0
test_score = 0.
start_time = time.clock()

epoch = 0
done_looping = False

while (epoch < n_epochs) and (not done_looping):
        print 'epoch = ', epoch
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter

            #print train_set_y.get_value(borrow=True).shape

            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
 
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    best_params = [layer0.params, layer2.params, layer3.params]

                    # test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of best '
                           'model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

end_time = time.clock()
print('Optimization complete.')
print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
print('Saving net.')
save_file = open(sys.argv[3], 'wb')
cPickle.dump(best_params[0][0].get_value(borrow=True), save_file, -1)
cPickle.dump(best_params[0][1].get_value(borrow=True), save_file, -1)
cPickle.dump(best_params[1][0].get_value(borrow=True), save_file, -1)
cPickle.dump(best_params[1][1].get_value(borrow=True), save_file, -1)
cPickle.dump(best_params[2][0].get_value(borrow=True), save_file, -1)
cPickle.dump(best_params[2][1].get_value(borrow=True), save_file, -1)
save_file.close()
