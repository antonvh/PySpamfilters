####### Build a dataset and dictionary for make spamfilters using machine learning ###########
#Based on exercise 6 of the coursera Machine Learning Course 

import os
import sf.helpers
import matplotlib.pyplot as plt

#settings
programdir = os.path.dirname(__file__)
dict_size = 1000
force_rebuild_dictionary =True


######Build training matrix#######

#check for a dictionary file
dictionary_file = os.path.join(programdir, 'mailwords.txt')
if os.path.isfile(dictionary_file) and not force_rebuild_dictionary:
    f = open(dictionary_file,'r')
    word_dict = eval(f.read())
    f.close
    rebuild_word_dict = False
else:
    rebuild_word_dict = True
    word_dict = {}
    
#spam and ham downloaded from http://spamassassin.apache.org/publiccorpus/
#files are in the data folder
spamdir = os.path.join(programdir, 'data/spam/')
hamdir = os.path.join(programdir, 'data/ham/')

#scrub emails and collect all words for a dictionary
(spamlist,word_dict) = sf.helpers.process_email_dir(spamdir,word_dict,rebuild_word_dict)
(hamlist,word_dict) = sf.helpers.process_email_dir(hamdir,word_dict,rebuild_word_dict)


#save dictionary to file#
#TODO: make this a MySQL database.
if rebuild_word_dict:
    print("The total number of words in all mails is {0}".format(len(word_dict)))
    sorted_word_dict = sorted(word_dict.items(), key=lambda word_dict: word_dict[1])[-dict_size:]
    
    sorted_word_dict_enumerated=[]
    #renumber the words
    for i in range(len(sorted_word_dict)):
        sorted_word_dict_enumerated += [[sorted_word_dict[i][0],i]]
    
    word_dict = dict(sorted_word_dict_enumerated)
    f = open(dictionary_file,'w')
    
    f.write(repr(word_dict)) #save only 10,000 most used words.
    f.close
    
    #plot the graph of words and their occurrences
    occurrences = [w[1] for w in sorted_word_dict[-dict_size:]]
    print "These are the 10 most occurring words are"
    print sorted_word_dict[-9:]
    plt.plot(occurrences)
    plt.xlabel('word indices')
    plt.ylabel('occurrences')
    plt.ylim([0,5000])
    plt.show()





######## Build training set and save to file ############
print "Saving to file..."
#PyBrain has some nice classes to do all this.
from pybrain.datasets import SupervisedDataSet
import numpy as np

DS = SupervisedDataSet(dict_size,1)

for m_list,target in [[spamlist,1],[hamlist,0]]:
    for mail in m_list:
        #each data point is a list (or vector) the size of the dictionary
        wordvector=np.zeros(dict_size)
        #now go through the email and put the occurrences of each word
        #in it's respective spot (i.e. word_dict[word]) in the vector 
        for word in mail:
            if word in word_dict:
                wordvector[word_dict[word]] += 1
        DS.appendLinked(np.log(wordvector+1)   , [target]) #put word occurrences on a log scale

#TODO: use MySQL instead of csv
DS.saveToFile('dataset.csv')
print "Done."

