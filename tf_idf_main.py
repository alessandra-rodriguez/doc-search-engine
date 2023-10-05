##########
## Import libraries
##########
import os
from search_functions import process_file, query

##########
## Initialize Variables
##########
corpusroot = ''

##########
## Driver Code
##########
if __name__ == "__main__":
    for filename in os.listdir(corpusroot):
        file = open(os.path.join(corpusroot, filename), "r", encoding='windows-1252')
        doc = file.read()
        file.close()
        process_file(filename, doc)


    print("(%s, %.12f)" % query("cool facts about space"))
    print("(%s, %.12f)" % query("american history"))