# TF-IDF Search Engine

## Description:
The following code implements a term frequency inverse document frequency search engine for a corpus of documents to query on. The documents in the corpus will be ranked by their normalized tf-idf scores and the most relevant document to the query will be returned. The following weighting scheme is used: ltc.lnc
* Document: logarithmic tf, logarithmic idf, cosine normalization
* Query: logarithmic tf, no idf, cosine normalization

## Features:
* Normalized TF-IDF scores for queries and documents
* Finds document name and score most relevant to query
* Tokenization, stop word removal, stemming

## Usage
1. Clone this repo locally
2. Install and update relevant libraries
3. Identify corpus directory and update 'corpusroot'
4. Use the provided functions to perform document searches based on a query