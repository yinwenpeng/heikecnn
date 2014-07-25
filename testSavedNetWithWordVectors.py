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

if len(sys.argv) != 4:
  print "please pass the saved network, the testing file and the word vectors as parameter"
  exit(0)

netfilename = sys.argv[1]
testfile = sys.argv[2]
vectorfile = sys.argv[3]

# reading word vectors
wordvectors = {}
vectorsize = 0
f = open(vectorfile, 'r')
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

contextsize = 18

# load test file
numSamplesTest = 0
f = open(testfile, "r+")
buf = mmap.mmap(f.fileno(), 0) # seems to be faster than normal file read for line count
readline = buf.readline
while readline():
  numSamplesTest += 1
f.close()

f = open(testfile, 'r')
inputMatrixTest = numpy.empty(shape = (numSamplesTest, vectorsize * contextsize))
resultVectorTest = []
sample = 0
for line in f:
  line = line.strip()
  parts = line.split(' : ')
  posNeg = parts[0]
  if "+" in posNeg:
    resultVectorTest.append(1)
  else:
    resultVectorTest.append(0)
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
  inputMatrixTest[sample,:] = matrix
  sample += 1
f.close()

##################### the network #######################
dt = theano.config.floatX
test_set_x = theano.shared(numpy.matrix(inputMatrixTest, dtype = dt))
test_set_y = theano.shared(numpy.array(resultVectorTest, dtype = numpy.dtype(numpy.int32)))

batch_size=1
nkerns=[20, 50]
rng = numpy.random.RandomState(23455)

n_test_batches = test_set_x.get_value(borrow=True).shape[0]
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
layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

# the cost we minimize during training is the NLL of the model
cost = layer3.negative_log_likelihood(y)

# create a function to compute the mistakes that are made by the model
test_model = theano.function([index], layer3.errors(y),
         givens={
             x: test_set_x[index * batch_size: (index + 1) * batch_size],
             y: test_set_y[index * batch_size: (index + 1) * batch_size]})

test_model_confidence = theano.function([index], layer3.results(y),
         givens={
             x: test_set_x[index * batch_size: (index + 1) * batch_size]})

# load parameters
netfile = open(netfilename)
layer0.params[0].set_value(cPickle.load(netfile), borrow=True)
layer0.params[1].set_value(cPickle.load(netfile), borrow=True)
layer2.params[0].set_value(cPickle.load(netfile), borrow=True)
layer2.params[1].set_value(cPickle.load(netfile), borrow=True)
layer3.params[0].set_value(cPickle.load(netfile), borrow=True)
layer3.params[1].set_value(cPickle.load(netfile), borrow=True)

# test net on test file
confidence = [test_model_confidence(i) for i in xrange(n_test_batches)]
#test_losses = [test_model(i) for i in xrange(n_test_batches)]
print "confidence, hypothesis, reference"
for index in range(0, len(confidence)): #zip(confidence, resultVectorTest):
  print "conf: " + str(confidence[index][1]) + " - hypo: " + str(confidence[index][0]) + " - ref: " + str(resultVectorTest[index])
  #print t
#test_score = numpy.mean(test_losses)
#print(('    test error: %f ') %
#     (test_score * 100.))
