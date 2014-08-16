from pybrain.datasets import SupervisedDataSet
print "Reading data set.."
DS = SupervisedDataSet.loadFromFile('dataset.csv')

#Split validation set
DStest, DStrain = DS.splitWithProportion( 0.25 )

#train nn
from sf.helpers import NeuralNet3L

print "Training network with {0} examples".format(len(DStrain))
net = NeuralNet3L(len(DStrain['input'][0]), 200, 1)
net.train(DStrain,lambda_reg=5,maxiter=40)


pvec = net.activate(DStest['input'])
err = 0
m = len(pvec)
print "Testing with {0} examples.".format(len(DStest))
for i in range(m):
    p = round(pvec[i])
    t = DStest['target'][i]
    if p != t:err+=1
    
print "Error on test set is:{0}%".format(err*100/m)
