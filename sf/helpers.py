import re
import os
from nltk.stem.porter import PorterStemmer
#from nltk.tokenize.regexp import RegexpTokenizer

def read_and_scrub(fileloc,word_dict,rebuild_word_dict=False):
    f = open(fileloc,'r')
    fstring = f.read()
    f.close()
    
    fstring=fstring.lower()
    
    #take header and body apart
    #get everything after the first two newlines.
    email_contents = re.split('\n\n',fstring,maxsplit=1)[1]

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.sub('<[^<>]+>', ' ',email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.sub('[0-9]+', ' number ',email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr',email_contents)

    # Handle Email Addresses
    #Look for strings with @ in the middle
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr',email_contents)

    # Handle $ sign
    email_contents = re.sub('[$]+', ' dollar ',email_contents)

    # Split in words and remove white space and punctuation
    email_word_list = re.split("""[]!"#$%&'()*+,\-./:;<=>?@^_`{|}~ \t\r\n\v\f\\\[]*""",email_contents)
    
    #email_wordcodes = []
    
    #unique_id=len(word_dict) #continue numbering unique id's here
    for i in reversed(range(len(email_word_list))):
        w = email_word_list[i]
        if len(w) > 1: #skip 1 character words or punctuation and skip bs words.
            w = PorterStemmer().stem(w)
            if len(w) > 20: #apparently it's a really long bs word. 
                w='really_long_bs_word'
            if rebuild_word_dict:
                    if w in word_dict:
                        word_dict[w] += 1
                    else:
                        word_dict[w] = 1
        else:
            email_word_list.pop(i)
                      
    return email_word_list,word_dict
    
def update_progress(progress):
    print '\r[{0}] {1}%'.format('#'*(progress/10), progress)
    
    
def process_email_dir(directory,word_dict,rebuild_word_dict):
    
    #Read spam files, scrub & stem
    
    files = [ f for f in os.listdir(directory) if (os.path.isfile(os.path.join(directory,f)) and f[0] <> ".") ]

    mail_list = []

    n=len(files)
    for i in range(n):
        f=files[i]
        (scrubbedmail,word_dict) = read_and_scrub(os.path.join(directory, f),word_dict,rebuild_word_dict)
        mail_list += [scrubbedmail]
        update_progress(i*100/n)
    return mail_list,word_dict

####my own neural network implementation###
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_gradient(z):
    sz = expit(z)
    return sz*(1-sz)

def rand_init_theta(n_input,n_out):
    epsilon_init = 0.12
    return (np.random.rand(n_out, 1 + n_input) * 2 * epsilon_init) - epsilon_init
    
class NeuralNet3L(object):
    def __init__(self,n_input,n_middle,n_out):
        self.n_input = n_input
        self.n_middle = n_middle
        self.n_out = n_out
        self.theta1 = rand_init_theta(n_input,n_middle)
        self.theta2 = rand_init_theta(n_middle,n_out)
    
    def activate_all_layers(self,X,theta1,theta2):
        # Create Bias matrix/array
        s = list(np.shape(X))
        s[-1]=1 #set the last dimension to 1
        Bias = np.ones(s)
            
        # Add ones to create the activations matrix of layer 1
        A1 = np.append(Bias,X,axis=1)

        # Run the first layer - compute activations
        Z2 = A1.dot(theta1.T)
        
        # Sigmoidize the result and add ones to create the activations matrix of layer 2
        A2 = np.append(Bias, expit(Z2),axis=1)

        # Run the second layer - compute activations
        Z3 = A2.dot(theta2.T)
        A3 = expit(Z3)
        
        return {'A1':A1,'A2':A2,'A3':A3,'Z2':Z2}
    
    def activate(self,X):
        return self.activate_all_layers(X,self.theta1,self.theta2)['A3']
    
    def nn_cost_func(self,theta_vec,DS,lambda_reg):
        theta1 = np.reshape(theta_vec[:self.theta1.size],self.theta1.shape)
        theta2 = np.reshape(theta_vec[self.theta1.size:],self.theta2.shape)
        
        activations=self.activate_all_layers(DS['input'],theta1,theta2)
        TargetA3 = DS['target'] #y or target activations
        m=len(DS)               #number of samples in test set
        #calculate cost
        J = sum(sum(-TargetA3*np.log(activations['A3'])-(1-TargetA3)*np.log(1-activations['A3'])))/m
        
        #regularize J
        J = J+lambda_reg/2/m*(sum(sum(theta1[:,1:]**2))+sum(sum(theta2[:,1:]**2)))
        
        return J
        
    def nn_grad_func(self,theta_vec,DS,lambda_reg):
        theta1 = np.reshape(theta_vec[:self.theta1.size],self.theta1.shape)
        theta2 = np.reshape(theta_vec[self.theta1.size:],self.theta2.shape)
        
        activations=self.activate_all_layers(DS['input'],theta1,theta2)
        TargetA3 = DS['target']
        m=len(DS) 
        
        #calculate gradient
        D3 = activations['A3']-TargetA3
        D2 = (D3.dot(theta2))[:,1:]*sigmoid_gradient(activations['Z2'])
        Theta2_grad = D3.T.dot(activations['A2'])/m
        Theta1_grad = D2.T.dot(activations['A1'])/m

        #regularize gradient
        Theta2Mask = np.ones(np.shape(self.theta2))
        Theta2Mask[:,0] = 0 # set the first column to zero
        Theta1Mask = np.ones(np.shape(self.theta1))
        Theta1Mask[:,0] = 0; # set the first column to zero

        Theta2_grad = Theta2_grad +lambda_reg/m*(theta2*Theta2Mask)
        Theta1_grad = Theta1_grad +lambda_reg/m*(theta1*Theta1Mask)
        
        # Unroll gradients
        grad = np.append(np.reshape(Theta1_grad,-1),np.reshape(Theta2_grad,-1))
        
        return grad
    
    def nn_cost_grad_func(self,theta_vec,DS,lambda_reg):
        theta1 = np.reshape(theta_vec[:self.theta1.size],self.theta1.shape)
        theta2 = np.reshape(theta_vec[self.theta1.size:],self.theta2.shape)
        
        activations=self.activate_all_layers(DS['input'],theta1,theta2)
        TargetA3 = DS['target'] #y or target activations
        m=len(DS)               #number of samples in test set
        #calculate cost
        J = sum(sum(-TargetA3*np.log(activations['A3'])-(1-TargetA3)*np.log(1-activations['A3'])))/m
        
        #regularize J
        J = J+lambda_reg/2/m*(sum(sum(theta1[:,1:]**2))+sum(sum(theta2[:,1:]**2)))
        
        #calculate gradient
        D3 = activations['A3']-TargetA3
        D2 = (D3.dot(theta2))[:,1:]*sigmoid_gradient(activations['Z2'])
        Theta2_grad = D3.T.dot(activations['A2'])/m
        Theta1_grad = D2.T.dot(activations['A1'])/m

        #regularize gradient
        Theta2Mask = np.ones(np.shape(self.theta2))
        Theta2Mask[:,0] = 0 # set the first column to zero
        Theta1Mask = np.ones(np.shape(self.theta1))
        Theta1Mask[:,0] = 0; # set the first column to zero

        Theta2_grad = Theta2_grad +lambda_reg/m*(theta2*Theta2Mask)
        Theta1_grad = Theta1_grad +lambda_reg/m*(theta1*Theta1Mask)
        
        # Unroll gradients
        grad = np.append(np.reshape(Theta1_grad,-1),np.reshape(Theta2_grad,-1))
        
        return J,grad
    
    def train(self,DS,lambda_reg=1,maxiter=10):
        np.seterr(all='print')
        #unroll initial theta
        start_theta_vec = np.append(
                                    np.reshape(self.theta1,-1),
                                    np.reshape(self.theta2,-1)
                                    )
        
        result = minimize(self.nn_cost_grad_func, 
                          x0=start_theta_vec,
                          args = (DS,lambda_reg),
                          method='cg', 
                          jac=True, #means we return a gradient in the cost function
                          options={'maxiter':maxiter,'disp':True}
                          ) 
        
        self.theta1 = np.reshape(result.x[:self.theta1.size],self.theta1.shape)
        self.theta2 = np.reshape(result.x[self.theta1.size:],self.theta2.shape)
        