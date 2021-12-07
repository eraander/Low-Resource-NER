import re
import os
import regex
import copy
import sys
import random
import numpy as np
import pandas as pd

from decimal import ROUND_HALF_UP, Context
from typing import Iterable, Sequence, Tuple, List, Dict, NamedTuple, Optional, Counter, Mapping
from pymagnitude import Magnitude
from collections import defaultdict

import nltk
from nltk.metrics import scores, ConfusionMatrix

import spacy
from spacy.tokens import Doc, Token, Span
from spacy.language import Language
from pycrfsuite import Tagger, Trainer, ItemSequence
import json

from cosi217.debug import print_ents
from cosi217.ingest import load_conll2003
from cosi217.ingest import spacy_doc_from_sentences
from hw3utils import EntityEncoder, PRF1
from hw3utils import FeatureExtractor, ScoringCounts, ScoringEntity
from hw3utils import UPPERCASE_RE, LOWERCASE_RE, DIGIT_RE, PUNC_REPEAT_RE
from cupy_utils import *
from embeddings import read


def ingest_json_document(doc_json: Mapping, nlp: Language) -> Doc:
    #raise error if not annotated (no annotator and labels are empty)
    if doc_json["annotation_approver"] == None and doc_json["labels"] == []:
        raise ValueError("not annotated")
    #text string from json
    text = doc_json["text"]
    #list labels w/char span in json
    labels = doc_json["labels"]
    #result doc to return
    doc = nlp(text)
    #get labels if there are entities labeled in file
    if labels != []:
        #tokens in doc.text
        tokens = [token for token in doc]
        #list of span objs to add to doc.ents
        ents = []
        #make ent and text before ent into docs and count tokens to
        #convert char span to token span
        for label in labels:
            char_range = doc.text[label[0]:label[1]]
            bef_char = doc.text[:label[0]]
            char_range = [token for token in nlp(char_range)]
            bef_char = [token for token in nlp(bef_char)]
            beg = len(bef_char)
            end = beg + len(char_range)
            lab = label[2][-3:]
            ents.append(Span(doc, beg, end, lab))
        #set ent list as doc ents           
        doc.ents = ents
    return doc

class WordVectorFeatureEng(FeatureExtractor):
    def __init__(self, vectors_path: str, scaling: float = 1.0) -> None:
        #word vectors
        self.vect = Magnitude(vectors_path,
normalized=False)
        #salar to multiply vector by
        self.scalar = scaling
        #list "v0"..."v"+size of vectors (vect.dim)
        self.v_keys = ["v" +str(i) for i in range(0, self.vect.dim)]

    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if relative_idx == 0:
            #scaled word vector for token(numpy array)
            v = self.vect.query(token)*self.scalar
            #feature dictionary {"v0":v[0], "v1":v[1],...} for token
            features.update(zip(self.v_keys,
v))

'''takes tuple of word list, numpy array of corresponding vectors
   adds features to feature dict for each number in vector for the word
   if the word is in the list of words
   Because it is meant to be used with sami/english embeddings from VecMap-
   assumes each word is preceded by "sme:" in the wordlist
'''
class WordVectorFeature(FeatureExtractor):
    def __init__(self, vectors: tuple, scaling: float = 1.0, length: int = 300) -> None:
        #word vectors
        self.vect = vectors
        #salar to multiply vector by
        self.scalar = scaling
        #list "v0"..."v"+size of vectors (vect.dim)
        self.v_keys = ["v" +str(i) for i in range(0, length)]

    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if relative_idx == 0:
            try:
                num = self.vect[0].index('sme:'+ token)
                #find corresponding vector and multiply by scalar
                v = self.vect[1][num] * self.scalar
                #feature dictionary {"v0":v[0], "v1":v[1],...} for token
                features.update(zip(self.v_keys, v))
            except ValueError:
                pass
                
'''checks given name list and adds to features if toke in namelist
'''
class NameListFeature(FeatureExtractor):
    def __init__(self, namelist: list):
        self.namelist = namelist
        
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        i = relative_idx+current_idx
        if i> -1:
            if i < len(tokens):
                if tokens[i] in self.namelist:
                    k = "name[" + str(relative_idx) + "]"
                    features[k] = 1.0
        
class BiasFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        features["bias"] = 1.0

class TokenFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        i = relative_idx+current_idx
        if i> -1:
            if i < len(tokens):
                k = "tok[" + str(relative_idx) + "]=" + tokens[i]
                features[k] = 1.0

'''Gets feature for suffix (last 2 chars)
   if token if token is >= 5 chars
'''
class SuffixFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        i = relative_idx+current_idx
        if i> -1:
            if i < len(tokens):
                if len(tokens[i]) >= 5: 
                    k = "tok[" + str(relative_idx) + "]=" + tokens[i][-2:]
                    features[k] = 1.0

'''tries to extract locative marker (-s/-in)
'''
class LocativeFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        i = relative_idx+current_idx
        if i> -1:
            if i < len(tokens) and len(tokens[i])>=2:
                if tokens[i][-1] == 's': 
                    k = "loc[" + str(relative_idx) + "]"
                    features[k] = 1.0
                elif tokens[i][-2] == 'in': 
                    k = "loc[" + str(relative_idx) + "]"
                    features[k] = 1.0
        
class UppercaseFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        i = relative_idx+current_idx
        if i> -1:
            if i < len(tokens):
                if tokens[i].isupper():
                    k = "uppercase[" + str(relative_idx) + "]"
                    features[k] = 1.0

class TitlecaseFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        i = relative_idx+current_idx
        if i> -1:
            if i < len(tokens):
                if tokens[i].istitle():
                    k = "titlecase[" + str(relative_idx) + "]"
                    features[k] = 1.0

#assumes tokens is one sentence, so tokens[0] is initial
class InitialTitlecaseFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        i = relative_idx+current_idx
        if i == 0:
            if tokens[i].istitle():
                k = "initialtitlecase[" + str(relative_idx) + "]"
                features[k] = 1.0

class PunctuationFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        i = relative_idx+current_idx
        if i> -1:
            if i < len(tokens):
                if PUNC_REPEAT_RE.match(tokens[i]):
                    k = "punc[" + str(relative_idx) + "]"
                    features[k] = 1.0

class DigitFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        i = relative_idx+current_idx
        if i> -1:
            if i < len(tokens):
                if DIGIT_RE.search(tokens[i]):
                    k = "digit[" + str(relative_idx) + "]"
                    features[k] = 1.0

class WordShapeFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        i = relative_idx+current_idx
        if i> -1:
            if i < len(tokens):
                s = tokens[i]
                s = LOWERCASE_RE.sub( 'x', s)
                s = UPPERCASE_RE.sub( 'X', s)
                s = DIGIT_RE.sub('0', s)
                k = "shape[" + str(relative_idx) + "]=" + s
                features[k] = 1.0

class WindowedTokenFeatureExtractor:
    def __init__(self, feature_extractors: Sequence[FeatureExtractor], window_size: int):
        self.extractors = feature_extractors
        self.window = window_size
        
        
    def extract(self, tokens: Sequence[str]) -> List[Dict[str, float]]:
        curr_index = 0
        features = []
        for token in tokens:
            feat_dict = {} #holds features for token
            for e in self.extractors:
                #get all features
                #feature extractors passed must exist
                w = 0-self.window
                actual_index = curr_index-self.window
                last = curr_index + self.window
                while actual_index <= last:
                    e.extract(token, curr_index, w, tokens, feat_dict)
                    w+=1
                    actual_index+= 1
            #add dict to features and increase index
            curr_index += 1
            features.append(feat_dict)
        return features

class CRFsuiteEntityRecognizer:
    def __init__(
        self, feature_extractor: WindowedTokenFeatureExtractor, encoder: EntityEncoder
    ) -> None:
        self.enc = encoder
        self.extractor = feature_extractor
        self.tagger = Tagger()
        self.train_called = False

    @property
    def encoder(self) -> EntityEncoder:
        return self.enc

    def train(self, docs: Iterable[Doc], algorithm: str, params: dict, path: str) -> None:
        self.train_called = True
        trainer = Trainer(algorithm, verbose=False)
        trainer.set_params(params)
        for doc in docs:
            for sent in doc.sents:
                tokens = list(sent)
                tokens_str = [str(i) for i in tokens]
                features = self.extractor.extract(tokens_str)
                labels = self.enc.encode(tokens)
                trainer.append(features, labels)
        trainer.train(path)
        self.tagger.open(path)

    def __call__(self, doc: Doc) -> Doc:
        if self.train_called == False:
            raise ValueError
        ents = []
        for sent in doc.sents:
            tokens = list(sent)
            tokens_str = [str(i) for i in tokens]
            tags = self.predict_labels(tokens_str)
            ents += decode_bilou(tags, tokens, doc)          
        doc.ents = ents
        return doc
        
    def predict_labels(self, tokens: Sequence[str]) -> List[str]:
        feat = self.extractor.extract(tokens)
        p = self.tagger.set(feat)
        labels = self.tagger.tag(p)
        return labels
    
class BILOUEncoder(EntityEncoder):
    #takes sequence of Tokens
    #returns list of BILOU labels
    def encode(self, tokens: Sequence[Token]) -> List[str]:
        labels = []
        bio = BIOEncoder().encode(tokens)
        length = len(bio)
        # counter
        i = 0
        while i < (length):
            # if the label begins with a B it will change to a U
            # if it is not followed by I label
            # otherwise it remains a B
            if bio[i][:1] == "B":
                if (i == length-1) or bio[i+1][:1] == "B" or bio[i+1][:1] == "O":
                    new_label = bio[i][1:]
                    labels.append("U"+ new_label)
                else:
                    labels.append(bio[i])
                i += 1
            # if label is an I it will change to L if it is at the end
            # of a sequence starting with B (BI*L)
            elif bio[i][:1] == "I":
                if (i == length-1) or bio[i+1][:1] != "I":
                    new_label = bio[i][1:]
                    labels.append("L"+ new_label)
                else: 
                    labels.append(bio[i])
                i += 1
            # if the label is O it will remain O
            else:
                labels.append(bio[i])
                i += 1 
        return labels

"""
uses abstact class Entity Encoder
used to get BIO labels
"""
class BIOEncoder(EntityEncoder):
    #takes sequence of Tokens
    #outputs list of BIO tokens
    def encode(self, tokens: Sequence[Token]) -> List[str]:
        tokens_str = [str(i) for i in tokens]
        labels = []
        for t in tokens:
            bio = t.ent_iob_
            if bio == "":
                labels.append("O")
            else:
                labels.append(bio + "-" + t.ent_type_)
        return labels

def decode_bilou(labels: Sequence[str], tokens: Sequence[Token], doc: Doc) -> List[Span]:
    result = []
    #put together labels and tokens into tuples
    labeled_data = list(zip(labels, tokens))
    l = len(labeled_data)
    count = 0
    #iterate over labeled_data
    while count < l:
        x = []
        # check if token is beginning of entity (see assignment for requirements)
        if ((labeled_data[count][0][0] == 'B')
                    or (labeled_data[count][0][0] == 'U')
                    or (count == 0 and labeled_data[count][0][0] != "O")
                    or (labeled_data[count][0][0]== 'I'  and labeled_data[count-1][0] == 'O')
                    or (labeled_data[count][0][0]== 'L'  and labeled_data[count-1][0] == 'O')
                    or (labeled_data[count][0][0]== 'I'  and labeled_data[count-1][0][-3:] != labeled_data[count][0][-3:])
                    or (labeled_data[count][0][0]== 'L'  and labeled_data[count-1][0][-3:] != labeled_data[count][0][-3:])):
            #append to list if entity is beginning
            x.append(labeled_data[count])
            count +=1
            #continue to append until entity is complete
            while (count< l and (labeled_data[count][0][-3:] == x[-1][0][-3:])
                   and not ((labeled_data[count][0][0] == 'B')
                    or (labeled_data[count][0][0] == 'U'))):
                x.append(labeled_data[count])
                count +=1
        #if there are non-"O" labels, create Span object and add to result list
        if len(x) != 0:
            result.append(Span(doc, x[0][1].i, x[-1][1].i+1, x[0][0][-3:]))
        #update count if it was only "O"
        else:
            count += 1
    return result

def span_scoring_counts(
    reference_docs: Sequence[Doc], test_docs: Sequence[Doc], typed: bool = True
) -> ScoringCounts:
    results = {}
    
    ref_dict = defaultdict(list)
    test_dict = defaultdict(list)
    #put all entitites from docs in dictionary
    #label:[(start, end, doc identifying #)]
    count = 0
    for doc in reference_docs:
        ref = doc.ents
        test = test_docs[count].ents
        for r in ref:
            label = r.label_
            tup = (r.start, r.end, count, r.text)
            ref_dict[label].append(tup)
        for t in test:
            label = t.label_
            tup = (t.start, t.end, count, t.text)
            test_dict[label].append(tup)
        count +=1
    
    #result= ScoringCounts(Counter(), Counter(), Counter())
    tp = []
    fn = []
    fp = []
    #got through dicts and count tp, fp, fn- rec in test, not ref,
    tlab = test_dict.keys()
    rlab = ref_dict.keys()
    if typed:
        for l in rlab:
            for spn in ref_dict[l]:
                if (spn in test_dict[l]):
                    tp.append(ScoringEntity(tuple(spn[3].split()), str(l)))
                else:
                    fn.append(ScoringEntity(tuple(spn[3].split()), str(l)))
        for l in tlab:
            for spn in test_dict[l]:
                if (spn not in ref_dict[l]):
                    fp.append(ScoringEntity(tuple(spn[3].split()), str(l)))
    else:
        r_lst = []
        t_lst = []
        for k in tlab:
            t_lst = t_lst+test_dict.get(k)
        for k in rlab:
            r_lst = r_lst+ ref_dict.get(k)
        for spn in r_lst:
                if (spn in t_lst):
                    tp.append(ScoringEntity(tuple(spn[3].split()), ""))
                else:
                    fn.append(ScoringEntity(tuple(spn[3].split()), ""))
        for spn in t_lst:
            if (spn not in r_lst):
                fp.append(ScoringEntity(tuple(spn[3].split()), ""))
            
    return ScoringCounts(Counter(tp), Counter(fp), Counter(fn))
                                             
                                              
def span_prf1(
    reference_docs: Sequence[Doc], test_docs: Sequence[Doc], typed: bool = True
) -> Dict[str, PRF1]:
    result = {}
    
    ref_dict = defaultdict(list)
    test_dict = defaultdict(list)
    #put all entitites from docs in dictionary
    #label:[(start, end, doc identifying #)]
    count = 0
    for doc in reference_docs:
        ref = doc.ents
        test = test_docs[count].ents
        for r in ref:
            label = r.label_
            tup = (r.start, r.end, count)
            ref_dict[label].append(tup)
        for t in test:
            label = t.label_
            tup = (t.start, t.end, count)
            test_dict[label].append(tup)
        count +=1

    
    if typed:
        totalref = []
        totaltest = []
        for i in ref_dict:
            for z in ref_dict[i]:
                totalref.append((i, z))
            p = 0.0
            r = 0.0
            f = 0.0
            if i in test_dict:
                ref_s = set(ref_dict[i])
                test_s = set(test_dict[i])
                p = scores.precision(ref_s, test_s)
                r = scores.recall(ref_s, test_s)
                f = scores.f_measure(ref_s, test_s)
            if p == None:
                p = 0.0
            if r == None:
                r = 0.0
            if f == None:
                f = 0.0
            result[i] = PRF1(p, r, f)
        for i in test_dict:
            for z in test_dict[i]:
                totaltest.append((i, z))
            if i not in ref_dict:
                result[i] = PRF1(0.0, 0.0, 0.0)
        test_s = set(totaltest)
        ref_s = set(totalref)
        if test_s != None and ref_s != None:
            p = scores.precision(ref_s, test_s)
            r = scores.recall(ref_s, test_s)
            f = scores.f_measure(ref_s, test_s)
            if p == None:
                p = 0.0
            if r == None:
                r = 0.0
            if f == None:
                f = 0.0
        result[""] = PRF1(p, r, f)
    
    else:
        ref = []
        test = []
        for i in ref_dict:
            ref+= ref_dict[i]
        for i in test_dict:
            test += test_dict[i]
        p = 0.0
        r = 0.0
        f = 0.0
        s_test = set(test)
        s_ref = set(ref)
        if s_test!= None and s_ref != None:
            p = scores.precision(s_ref, s_test)
            r = scores.recall(s_ref, s_test)
            f = scores.f_measure(s_ref, s_test)
            if p == None:
                p = 0.0
            if r == None:
                r = 0.0
            if f == None:
                f = 0.0
        result[""] = PRF1(p, r, f)

    return result

def span_prf1_type_map(
    reference_docs: Sequence[Doc],
    test_docs: Sequence[Doc],
    type_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, PRF1]:

    result = {}

    ref_dict = defaultdict(list)
    test_dict = defaultdict(list)
    #put all entitites from docs in dictionary
    #label:[(start, end, doc identifying #)]
    count = 0
    for doc in reference_docs:
        ref = doc.ents
        test = test_docs[count].ents
        for r in ref:
            label = r.label_
            tup = (r.start, r.end, count)
            if (type_map != None) and (label in type_map):
                ref_dict[type_map[label]].append(tup)
            else:
                ref_dict[label].append(tup)
        for t in test:
            label = t.label_
            tup = (t.start, t.end, count)
            if (type_map != None) and (label in type_map):
                test_dict[type_map[label]].append(tup)
            else:
                test_dict[label].append(tup)
        count +=1
    
    totalref = []
    totaltest = []
    for i in ref_dict:
        for z in ref_dict[i]:
            totalref.append((i, z))
        p = 0.0
        r = 0.0
        f = 0.0
        if i in test_dict:
            ref_s = set(ref_dict[i])
            test_s = set(test_dict[i])
            p = scores.precision(ref_s, test_s)
            r = scores.recall(ref_s, test_s)
            f = scores.f_measure(ref_s, test_s)
        if p == None:
            p = 0.0
        if r == None:
            r = 0.0
        if f == None:
            f = 0.0
        result[i] = PRF1(p, r, f)
    for i in test_dict:
        for z in test_dict[i]:
            totaltest.append((i, z))
        if i not in ref_dict:
            result[i] = PRF1(0.0, 0.0, 0.0)
    test_s = set(totaltest)
    ref_s = set(totalref)
    if test_s != None and ref_s != None:
        p = scores.precision(ref_s, test_s)
        r = scores.recall(ref_s, test_s)
        f = scores.f_measure(ref_s, test_s)
        if p == None:
            p = 0.0
        if r == None:
            r = 0.0
        if f == None:
            f = 0.0
    result[""] = PRF1(p, r, f)
    return result

'''reads in text file with list of names
   returns list of names (to be used w/namelist feature)
'''
def getNameList(filename):
    f= open(filename)
    file = f.readlines()
    #get namelist and add all names to list
    return [n.split()[0] for n in file]

'''reads in vector text file
   returns tuple ([list of words], numpy array of vectors corresponding to words)
   uses read from embeddings.py--> comes with sami embeddings from VecMap - open source
   https://github.com/artetxem/vecmap
'''
def getVectors(filename):
    f = open(filename)
    return read(f)

def addPRF(a:PRF1, b:PRF1) -> PRF1:
    p = a.precision + b.precision
    r = a.recall + b.recall
    f = a.f1 + b.f1
    return PRF1(p, r, f)

def divPRF(x:PRF1, num:float):
    p = x.precision/num
    r = x.recall/num
    f = x.f1/num
    return PRF1(p, r, f)

def main_sami():
    NLP = spacy.load("en_core_web_sm", disable=["ner", "tagger"])
    
    
    filename = 'Northern_Sami.jsonl'
    all_docs = []
    with open(filename) as f:
        file = f.readlines()
        for l in file:
            my_doc = json.loads(l)
            doc = None
            err = False
            try:
                doc = ingest_json_document(my_doc, NLP)
            except (ValueError, IndexError):
                err = True
                continue
            if err == False:
                all_docs.append(doc)
    
    
    
    #crf using windowed token features
    crf = CRFsuiteEntityRecognizer(
        WindowedTokenFeatureExtractor(
            [
                BiasFeature(),
                TokenFeature(),
                UppercaseFeature(),
                #TitlecaseFeature(),
                #InitialTitlecaseFeature(),
                #PunctuationFeature(),
                #DigitFeature(),
                WordShapeFeature(),
                #LocativeFeature(),
                SuffixFeature(),
                NameListFeature(getNameList('northern_sami_given_names.txt')),
                #NameListFeature(getNameList('finnish_given_names.txt')),
                WordVectorFeature(getVectors('model_sme_fin_no.vec'), 0.5),  #sami- english embeddings
                #WordVectorFeature(getVectors('model_sme_eng.vec'), 2.0), #sami-finnish embeddings
                
            ],
            2,
        ),
        BILOUEncoder(),#ENCODING- change encoder here
    )

    #shuffle docs
    random.shuffle(all_docs)
    #cross-validation k = 5
    folds = [all_docs[:10], all_docs[10:20], all_docs[20:30], all_docs[30:40], all_docs[40:50]]
    folds2 = copy.deepcopy(folds)
    for i in folds2:
        for doc in i:
            doc.ents = []
    
    scores = None
    k = len(folds)-1
    count = k
    while count >= 0:
        train = []
        test_gold = folds[count]
        test = folds2[count]
        for i in range(k):
            if i != count:
                train += folds[i]

        #train model
        crf.train(train, "ap", {"max_iterations": 40}, "tmp.model")
        #test model
        test_results = [crf(doc) for doc in test]
        #update scores
        s = span_prf1(test_gold, test, typed=True)
        if count == k:
            scores = s
        else:
            for label in scores:
                scores[label] = addPRF(scores[label], s[label])
        count -=1
    #average scores
    for label in scores:
        scores[label] = divPRF(scores[label], k+1)

    print("=======N. Sami Results=======")
    print("Type\tPrec\tRec\tF1", file=sys.stderr)
    # Always round .5 up, not towards even numbers as is the default
    rounder = Context(rounding=ROUND_HALF_UP, prec=4)

    for ent_type, score in sorted(scores.items()):
        if ent_type == "":
            ent_type = "ALL"

        fields = [ent_type] + [
            str(rounder.create_decimal_from_float(num * 100)) for num in score
        ]
        print("\t".join(fields), file=sys.stderr)
        
def main_engl():
    NLP = spacy.load("en_core_web_sm", disable=["ner", "tagger"])
    
    #crf using windowed token features
    crf = CRFsuiteEntityRecognizer(
        WindowedTokenFeatureExtractor(
            [
                BiasFeature(),
                TokenFeature(),
                UppercaseFeature(),
                #TitlecaseFeature(),
                #InitialTitlecaseFeature(),
                #PunctuationFeature(),
                #DigitFeature(),
                WordShapeFeature(),
                #LocativeFeature(),
                SuffixFeature(),
                WordVectorFeatureEng('wiki-news-300d-1M-subword.magnitude', 2.0),
            ],
            2,
        ),
        BILOUEncoder(),#ENCODING- change encoder here
    )

    
    all_docs = load_conll2003("data/conll2003/en/valid.txt", NLP)
    #shuffle docs
    random.shuffle(all_docs)
    #cross-validation k = 5
    folds = [all_docs[:10], all_docs[10:20], all_docs[20:30], all_docs[30:40], all_docs[40:50]]
    folds2 = copy.deepcopy(folds)
    for i in folds2:
        for doc in i:
            doc.ents = []
    
    scores = None
    k = len(folds)-1
    count = k
    while count >= 0:
        train = []
        test_gold = folds[count]
        test = folds2[count]
        for i in range(k):
            if i != count:
                train += folds[i]

        #train model
        crf.train(train, "ap", {"max_iterations": 40}, "tmp.model")
        #test model
        test_results = [crf(doc) for doc in test]
        #update scores
        s = span_prf1(test_gold, test, typed=True)
        if count == k:
            scores = s
        else:
            for label in scores:
                scores[label] = addPRF(scores[label], s[label])
        count -=1
    #average scores
    for label in scores:
        scores[label] = divPRF(scores[label], k+1)
    print("=======English Results=======")
    print("Type\tPrec\tRec\tF1", file=sys.stderr)
    # Always round .5 up, not towards even numbers as is the default
    rounder = Context(rounding=ROUND_HALF_UP, prec=4)

    for ent_type, score in sorted(scores.items()):
        if ent_type == "":
            ent_type = "ALL"

        fields = [ent_type] + [
            str(rounder.create_decimal_from_float(num * 100)) for num in score
        ]
        print("\t".join(fields), file=sys.stderr)      
                
if __name__ == '__main__':
    main_sami()
    #main_engl()
