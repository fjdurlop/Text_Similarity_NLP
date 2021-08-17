"""
Modulo utilizado para el proyecto final de Procesamiento de Lenguaje Natural: 

Facultad de Ingenieria
UNAM
Francisco Javier Duran Lopez

Corpus: https://www.kaggle.com/abisheksudarshan/topic-modeling-for-research-articles
"""


import numpy as np
import pandas as pd
import nltk
from nltk.corpus import gutenberg
from datetime import datetime
import re
import matplotlib.pyplot as plt
import unicodedata
from contractions import CONTRACTION_MAP
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.toktok import ToktokTokenizer
import pdfplumber
from importlib import reload


def get_text_from_txt(txt_path):
    with open(txt_path, encoding="utf8") as file:
        text = file.read()
    return text

def get_text_from_pdf(pdf_path):
    """
    Get text from pdf, it doesn't consider two columns papers, still a problem

    Params:
        pdf_path(str): PDF path

    Returns:
        text(str): A string with the text of the PDF
    """
    raw_string = r"{}".format(pdf_path)
    try:
        with pdfplumber.open(raw_string) as pdf:
            text = ''
            text = [page.extract_text()+'\n' for page in pdf.pages]
            text = ''.join(text)
            return text
    except:
        print("There's a problem opening your file, check the path")


def corpus_to_df(docs_list:list)->pd.DataFrame:
    corpus = np.array(docs_list)
    return  pd.DataFrame({'Document': corpus})


def normalize_corpus(corpus, html_stripping=False, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True, 
                     text_lemmatization=True, special_char_removal=True, 
                     stopword_removal=True, remove_digits=True):
    
    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:
        # strip HTML
        #if html_stripping:
        #    doc = strip_html_tags(doc)
        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)
        # expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
        # lowercase the text    
        if text_lower_case:
            doc = doc.lower()
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
        # lemmatize text
        if text_lemmatization:
            doc = lem(doc)
        # remove special characters and\or digits    
        if special_char_removal:
            # insert spaces between special characters to isolate them    
            special_char_pattern = re.compile(r'([{._(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)  
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)
            
        normalized_corpus.append(doc)
        
    return normalized_corpus

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

 

# Init the Wordnet Lemmatizer

def lem(text):
    lemmatizer = WordNetLemmatizer()
    # Tokenize: Split the sentence into words
    word_list = nltk.word_tokenize(text)
    #print(word_list)
    #> ['The', 'striped', 'bats', 'are', 'hanging', 'on', 'their', 'feet', 'for', 'best']

    # Lemmatize list of words and join
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    #print(lemmatized_output)
    return lemmatized_output


tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    """Function to remove the stopwords
    """
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


def get_topics_of_doc(topics:pd.DataFrame,doc_idx:int)-> list:
    """
    Get a list of the topics of a doc.

    Dataset used in project from: https://www.kaggle.com/abisheksudarshan/topic-modeling-for-research-articles
    """
    return [topic for topic in topics.columns if topics.loc[doc_idx,topic]==1]

def num_of_coincidances(similar_papers_idxs:list,original_paper_idx:int,topics):
    """
    Number of total coincidances of topics of each document(similar_papers_idx) with the original paper

    Dataset used in project: https://www.kaggle.com/abisheksudarshan/topic-modeling-for-research-articles
    """
    count = 0
    for idx in similar_papers_idxs:
        #val = topics.iloc[idx]["Physics"]+topics.iloc[idx]["Cosmology and Nongalactic Astrophysics"]+topics.iloc[idx]["Instrumentation and Methods for Astrophysics"]
        val = 0
        list_topics = [topics.iloc[idx][topic] for topic in get_topics_of_doc(topics,original_paper_idx)]   
        if (sum(list_topics)>0): val=1
        #val += val
        count+=val
        #print(val)
    print(count)
    return count