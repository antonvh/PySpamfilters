from pybrain.datasets import ClassificationDataSet
print "Reading data set.."
DS = ClassificationDataSet.loadFromFile('dataset.csv')

#Split validation set
DStest, DStrain = DS.splitWithProportion( 0.25 )

#train nn
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer, sigmoidlayer
from pybrain.utilities           import percentError

net = buildNetwork(len(DStrain['input'][0]), 50, 50, 1, outclass=SoftmaxLayer)
#the softmax function, or normalized exponential,[1]:198 is a generalization of
# the logistic function that "squashes" a K-dimensional vector of arbitrary real
# values to a K-dimensional vector of real values in the range (0, 1)

trainer = BackpropTrainer(net, DStrain)
print "Training neural network..."
print "Training error after 1 epoch is:{0}".format(trainer.train())
print "Training error after 5 epochs is:{0}".format(trainer.trainEpochs(5))
#print "Training error until convergence is:{0}".format(trainer.trainUntilConvergence(dataset=DS,validationProportion=0.25))
trnresult = percentError(trainer.testOnClassData(),DStrain['class'] )
tstresult = percentError( trainer.testOnClassData(
           dataset=DStest ), DStrain['class'] )
print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult

print "Testing neural network..."
err=0          
for inp,targ in DStest:
    r = net.activate(inp)
    print "{0}, {1}".format(r,targ)
    if r != targ:
        err+=1
        
print "Error on test set is {0}%".format(err*100/len(DStest))