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
    
