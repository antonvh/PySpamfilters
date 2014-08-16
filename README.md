PySpamfilters
=============

Playing around with spam filters and machine learning

Get spam from the spamassissin public corpus

Put spam in the subfolder data/spam
Ham in data/ham
Remove any non-mail files from the download.

Needed python libraries for this code
* PyBrain
* Scipy
* Numpy
* LibSVM
* nltk
* matplotlib

There are three solutions to the spamfilter:
* train_svm - spamfilter with Support vector machine
* train_3L_nn - spamfilter using neural net from coursera machine learning
* train_pybrain_nn - spamfilter using pybrain buildnetwork. Doesn't work.