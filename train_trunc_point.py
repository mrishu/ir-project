#!/usr/bin/env python3
from tf_normalization_aspect import ritf, lrtf
import estimate_truncpt

import re
import xml.etree.ElementTree as ET
import seaborn as sns
import matplotlib.pyplot as plt

import lucene
from java.io import File

from org.apache.lucene.store import FSDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.index import Term
from org.apache.lucene.search import TermQuery
from org.apache.lucene.util import BytesRef

lucene.initVM()

analyzer = StandardAnalyzer()
indexPath = File("index-dir/").toPath()
directory = FSDirectory.open(indexPath)
reader = DirectoryReader.open(directory)  # index reader
termVecReader = reader.termVectors()
searcher = IndexSearcher(reader)
storedFields = searcher.storedFields()


def avgFieldLength(collectionStats):
    return collectionStats.sumTotalTermFreq() / collectionStats.docCount()


# Given DOCNO, returns the document's docid in the index
# Returns -1 if document with given DOCNO is not indexed
def docnoTodocid(docno):
    query = TermQuery(Term("DOCNO", docno))
    hits = searcher.search(query, 1).scoreDocs
    if len(hits) == 0:
        return -1
    return hits[0].doc


# Returns the tf's of all the terms of the query in document doc (docid)
def query_ritfs(raw_query, field, docid):
    termVec = termVecReader.get(docid, field)
    avgtf = termVec.getSumTotalTermFreq() / termVec.size()
    # remove special characters from raw_query except . and whitespace
    raw_query = re.sub('[^A-Za-z0-9.\\s]+', '', raw_query)
    query = QueryParser(field, analyzer).parse(raw_query).toString(field).split()
    termEnums = termVec.iterator()
    ritfs = []
    for term in query:
        if termEnums.seekExact(BytesRef(term)):
            tf = termEnums.totalTermFreq()
            ritfs.append(ritf(tf, avgtf))
    return ritfs


def query_lrtfs(raw_query, field, docid):
    collectionStats = searcher.collectionStatistics(field)
    termVec = termVecReader.get(docid, field)
    dl = termVec.getSumTotalTermFreq()
    avdl = avgFieldLength(collectionStats)
    # remove special characters from raw_query except . and whitespace
    raw_query = re.sub('[^A-Za-z0-9.\\s]+', '', raw_query)
    query = QueryParser(field, analyzer).parse(raw_query).toString(field).split()
    termEnums = termVec.iterator()
    lrtfs = []
    for term in query:
        if termEnums.seekExact(BytesRef(term)):
            tf = termEnums.totalTermFreq()
            lrtfs.append(lrtf(tf, avdl, dl))
    return lrtfs


robust_relavance_file = open('qrels/robust_601-700.qrel')
trec678_relavance_file = open('qrels/trec678_301-450.qrel')
robust_relavance_list = robust_relavance_file.readlines()
trec678_relavance_list = trec678_relavance_file.readlines()


# Given query number (int), return the relevant documents' docnos
def getRelDocs(query_num):
    relDocs = []

    if 601 <= query_num <= 700:
        for line in robust_relavance_list:
            ll = line.split()
            num, docno, relevance = int(ll[0]), ll[2], int(ll[3])
            if num == query_num and relevance != 0:
                relDocs.append(docno)

    elif 301 <= query_num <= 450:
        for line in trec678_relavance_list:
            ll = line.split()
            num, docno, relevance = int(ll[0]), ll[2], int(ll[3])
            if num == query_num and relevance != 0:
                relDocs.append(docno)

    return relDocs


robust_topics = ET.parse('topics/robust.xml').getroot()
trec678_topics = ET.parse('topics/trec678.xml').getroot()

relevant_ritfs = []
relevant_lrtfs = []

print("Training truncation point on the queries:")
for top in robust_topics[:10]:
    query_num = int(top[0].text)  # this is query number
    title = top[1].text  # this will be our query
    relDocs = getRelDocs(query_num)  # docnos of relevant documents of the query
    print(query_num, title)
    for relDoc in relDocs:
        docid = docnoTodocid(relDoc)
        if docid == -1:  # if the given relevant document isn't indexed
            continue
        relevant_ritfs = relevant_ritfs + query_ritfs(title, "TEXT", docid)
        relevant_lrtfs = relevant_lrtfs + query_lrtfs(title, "TEXT", docid)

for top in trec678_topics[:10]:
    query_num = int(top[0].text.strip())  # this is query number
    title = top[1].text.strip()  # this will be our query
    relDocs = getRelDocs(query_num)  # docno's of relevant documents of the query
    print(query_num, title)
    for relDoc in relDocs:
        docid = docnoTodocid(relDoc)
        if docid == -1:  # if the given relevant document isn't indexed
            continue
        relevant_ritfs = relevant_ritfs + query_ritfs(title, "TEXT", docid)
        relevant_lrtfs = relevant_lrtfs + query_lrtfs(title, "TEXT", docid)

print("\nMean RITF for relevant documents =", sum(relevant_ritfs) / len(relevant_ritfs))
print("Mean LRTF for relevant documents =", sum(relevant_lrtfs) / len(relevant_lrtfs))

truncpt_ritf_exp = estimate_truncpt.truncpt_EXP_T(relevant_ritfs, 0.1)
truncpt_lrtf_exp = estimate_truncpt.truncpt_EXP_T(relevant_lrtfs, 0.01)
# truncpt_ritf_gll = estimate_truncpt.truncpt_GLL_T(relevant_ritfs, 0.2)
# truncpt_lrtf_gll = estimate_truncpt.truncpt_GLL_T(relevant_lrtfs, 0.2)

print("\ntau1 assuming exponential distribution:", truncpt_ritf_exp)
print("tau2 assuming exponential distribution:", truncpt_lrtf_exp)
# print("tau1 assuming log-logistic distribution:", truncpt_ritf_gll)
# print("tau2 assuming log-logistic distribution:", truncpt_lrtf_gll)

fig, (ax1, ax2) = plt.subplots(1, 2)
sns.kdeplot(relevant_ritfs, ax=ax1)
sns.kdeplot(relevant_lrtfs, ax=ax2)

ax1.axvline(truncpt_ritf_exp, color='red', label='exp_tau1')
ax2.axvline(truncpt_lrtf_exp, color='red', label='exp_tau2')
# ax1.axvline(truncpt_ritf_gll, color='orange')
# ax2.axvline(truncpt_lrtf_gll, color='orange')

ax1.legend()
ax2.legend()
plt.show()

robust_relavance_file.close()
trec678_relavance_file.close()
