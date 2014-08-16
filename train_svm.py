from pybrain.datasets import ClassificationDataSet
print "Reading data set..."
DS = ClassificationDataSet.loadFromFile('dataset.csv')

#Split validation set
TestDS, TrainDS = DS.splitWithProportion( 0.25 )

#train svm
from svm import svm_problem, svm_parameter, libsvm, gen_svm_nodearray

#define problem with data from the pybrain dataset.
# best python explanation for libsvm is here: https://github.com/arnaudsj/libsvm/tree/master/python
#we have to convert the data to ints and lists because of the low-level c interface

prob = svm_problem([int(t) for t in TrainDS['target']],[list(i) for i in TrainDS['input']])
param = svm_parameter()
# option: -t 0: linear kernel. Best for classification.
# option: -c 0.01: regularization parameter. smaller is more regularization
# see below for all options
param.parse_options('-t 0 -c 0.01') 
print "Training svm..."
model = libsvm.svm_train(prob,param)

print "Testing svm with three random inputs"
from random import randrange
for j in range(3):
    i = randrange(0,len(TestDS))
    #again some conversion needed because of low level interface
    x0,m_idx = gen_svm_nodearray(list(TestDS['input'][i]))
    prediction = libsvm.svm_predict(model, x0)
    print("Target:{0}, prediction:{1}".format(TestDS['target'][i],prediction))
    
#test svm over test dataset
correct = 0
for j in range(len(TestDS)):
    #again some conversion needed because of low level interface
    x0,m_idx = gen_svm_nodearray(list(TestDS['input'][j]))
    prediction = libsvm.svm_predict(model, x0)
    if int(prediction) == int(TestDS['target'][j]):
        correct +=1
print "Accuracy on test set is {0}%".format(correct*100.0/len(TestDS))


###possible parameters###
# options:
# -s svm_type : set type of SVM (default 0)
#     0 -- C-SVC
#     1 -- nu-SVC
#     2 -- one-class SVM
#     3 -- epsilon-SVR
#     4 -- nu-SVR
# -t kernel_type : set type of kernel function (default 2)
#     0 -- linear: u'*v
#     1 -- polynomial: (gamma*u'*v + coef0)^degree
#     2 -- radial basis function: exp(-gamma*|u-v|^2)
#     3 -- sigmoid: tanh(gamma*u'*v + coef0)
# -d degree : set degree in kernel function (default 3)
# -g gamma : set gamma in kernel function (default 1/num_features)
# -r coef0 : set coef0 in kernel function (default 0)
# -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
# -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
# -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
# -m cachesize : set cache memory size in MB (default 100)
# -e epsilon : set tolerance of termination criterion (default 0.001)
# -h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
# -b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
# -wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)
