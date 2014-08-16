from pybrain.datasets import SupervisedDataSet
print "Reading data set.."
DS = SupervisedDataSet.loadFromFile('dataset.csv')

#Split validation set
DStest, DStrain = DS.splitWithProportion( 0.5 )

#train nn
from sf.helpers import NeuralNet3L

print "Training network"
net = NeuralNet3L(len(DStrain['input'][0]), 200, 1)
net.train(DStrain,lambda_reg=10,maxiter=10)
pvec = net.activate(DStest['input'])

err = 0
m = len(pvec)
for i in range(m):
    p = round(pvec[i])
    t = DStest['target'][i]
    if p != t:err+=1
    print "{0}, {1}".format(p,t)
    
print "Error on test set is:{0}%".format(err*100/m)
