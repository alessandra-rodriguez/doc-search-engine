##########
## Import libraries
##########
import math
import os
import nltk
import ssl
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
global tokenizer, stopWords, stemmer, word_counts, idf_cache, edited_files

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
nltk.download('stopwords')
stopWords = set(stopwords.words('english'))
stemmer = PorterStemmer()
edited_files = {}
idf_cache = {}
word_counts = {}

##########
## Functions
##########

"""
process_file: Uniformly tokenize and stem corpus files,
add processed files to a dictionary with filename as key and the processed file as a list of stemmed tokens as the value
add word count dictionaries to a dictionary with filename as the key and the word count dictionaries as the value

Parameters:
- file_name (string): full name of file including its extension
- file_content (string): contents of file as a string

Returns: Void. 
Updates two dictionaries:
    - 'edited_files' dictionary: Contains processed files with filenames as keys and lists of stemmed tokens as values.
    - 'word_counts' dictionary: Contains word count dictionaries with filenames as keys and word count dictionaries as values.
"""
def process_file(file_name, file_content):
    file_content = file_content.lower()
    file_content = tokenizer.tokenize(file_content)
    file_content = [stemmer.stem(word) for word in file_content if word not in stopWords]
    edited_files[file_name] = file_content
    word_counts[file_name] = get_word_count(file_content)

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
- file_name (string): name of file to find token weight
- token (string): word to find the weight of

Returns:
- normalized-tf-idf (double): weight of a token in a document
"""
def get_weight(file_name, token):
    word_dict = word_counts[file_name]
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
