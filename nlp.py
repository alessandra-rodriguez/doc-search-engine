##########
## Import libraries
##########
import os
import nltk
import ssl
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import math

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

##########
## Functions
##########

"""
get_word_count: Create dictionary for term frequencies of a document

Parameters:
- file (list): file as a list of tokens

Returns:
- dictionary: The area of the rectangle, calculated as length * width.
"""
def get_word_count(file):
    word_dict = {}
    for word in file:
        if word in word_dict:
            word_dict[word] += 1
        else:
            word_dict[word] = 1
    return word_dict

"""
get_idf: find the inverse document frequency of a token

Parameters:
- token (string): token of a document

Returns:
- idf (double): inverse doc. freq of a token, or -1 if token not in any docs
"""
def get_idf(token):
    if token in idf_cache.keys():
        return idf_cache[token]
    doc_count = 0
    for k in word_counts.values():
        if token in k.keys():
            doc_count += 1
    if doc_count > 0:
        idf = math.log(len(edited_files)/doc_count, 10)
        idf_cache[token] = idf
        return idf
    else:
        idf_cache[token] = -1
        return -1

"""
get_weight: find the normalized tf-idf of a token in a document

Parameters:
- filename (string): name of file to find token weight
- token (string): word to find the weight of

Returns:
- normalized-tf-idf (double): weight of a token in a document
"""
def get_weight(filename, token):
    word_dict = word_counts[filename]
    token = stemmer.stem(token)
    if token in word_dict.keys():
        weight_dict = {}
        for word in word_dict.keys():
            idf = get_idf(word)
            weight_dict[word] = (1 + math.log(word_dict[word], 10)) * idf
        norm = 1/(math.sqrt(sum([weight**2 for weight in weight_dict.values()])))
        return(weight_dict[token]*norm)
    else:
        return 0

"""
query: finds the most relevant document based on query 

Parameters:
- qstring (string): query used to find relevant documents

Returns:
- top_file_name, top_score (tuple): name and score of most similar document to query
"""
def query(qstring):
    top_score = -float('inf')
    top_file_name = None

    qstring = qstring.lower()
    qstring = tokenizer.tokenize(qstring)
    qstring = " ".join([word for word in qstring if word not in stopWords])

    query_dict = get_word_count(qstring.split())
    for key in query_dict.keys():
        query_dict[key] = (1 + math.log(query_dict[key], 10))

    norm = 1/math.sqrt(sum([(count**2) for count in query_dict.values()]))

    for key in query_dict.keys():
        query_dict[key] = query_dict[key] * norm


    for doc_name in edited_files.keys():
        query_token_weights = [get_weight(doc_name, token) for token in qstring.split()]
        query_score = sum([query_token_weights[i]*list(query_dict.values())[i] for i in range(len(query_token_weights))])

        if query_score > top_score:
            top_score = query_score
            top_file_name = doc_name

    return (top_file_name, top_score)

##########
## Initialize Variables
##########

corpusroot = ''
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
nltk.download('stopwords')
stopWords = set(stopwords.words('english'))
stemmer = PorterStemmer()
edited_files = {}
idf_cache = {}
word_counts = {}

##########
## Driver Code
##########

for filename in os.listdir(corpusroot):
    if filename.startswith('0') or filename.startswith('1'):
        file = open(os.path.join(corpusroot, filename), "r", encoding='windows-1252')
        doc = file.read()
        file.close()
        doc = doc.lower()
        doc = tokenizer.tokenize(doc)
        doc = [stemmer.stem(word) for word in doc if word not in stopWords]
        edited_files[filename] = doc
        word_counts[filename] = get_word_count(doc)


print("(%s, %.12f)" % query("cool facts about space"))
print("(%s, %.12f)" % query("american history"))