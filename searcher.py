#!/usr/bin/env python3
from tf_normalization_aspect import ritf, lrtf
from estimate_parameter import MLE_lambda_EXP_T
# from estimate_parameter import MLE_k_GLL_T
from distributions import EXP_T
# from distributions import GLL_T

import lucene
import re
import math

from java.io import File

from org.apache.lucene.store import FSDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import BooleanQuery
from org.apache.lucene.search import BooleanClause
from org.apache.lucene.search.similarities import BooleanSimilarity
from org.apache.lucene.index import Term
from org.apache.lucene.search import TermQuery
from org.apache.lucene.util import BytesRef

lucene.initVM()

TERM_MAX_POSTING_LIST = 5000
QUERY_MAX_POSTING_LIST = 500

indexPath = File("index-dir/").toPath()
analyzer = StandardAnalyzer()
directory = FSDirectory.open(indexPath)
reader = DirectoryReader.open(directory)
searcher = IndexSearcher(reader)
# using BooleanSimilarity is same as retrieving posting lists
searcher.setSimilarity(BooleanSimilarity())
storedFields = searcher.storedFields()
termVecReader = reader.termVectors()


def iDF(docFreq, docCount):
    return math.log(docCount / docFreq)


def avgFieldLength(collectionStats):
    return collectionStats.sumTotalTermFreq() / collectionStats.docCount()


class EXP_T_Scorer():
    def __init__(self, query, field, tau, alpha=0.5):
        self.raw_query = query
        self.field = field
        self.alpha = alpha
        self.tau = tau
        self.query = self._parse_query()
        self.collectionStats = searcher.collectionStatistics(field)
        self.avgdl = avgFieldLength(self.collectionStats)
        self.docCount = self.collectionStats.docCount()
        self.tf_sample_list = []
        self.docFreq_list = []
        for term in self.query.split():
            ritf_samples, lrtf_samples, docFreq = self._normalized_tf_sampling_and_docFreq(term)
            self.tf_sample_list.append((ritf_samples, lrtf_samples))
            self.docFreq_list.append(docFreq)

    # Returns a parsed query which is all the terms separeted by space
    # and passed through the analyzer
    def _parse_query(self):
        # remove special characters from raw_query
        # except . and whitespace before passing to QueryParser
        raw_query = re.sub('[^A-Za-z0-9.\\s]+', '', self.raw_query)
        query = QueryParser(self.field, analyzer).parse(raw_query).toString(self.field)
        return query

    def _query_interesection_list(self):
        boolqBuilder = BooleanQuery.Builder()
        for term in self.query.split():
            boolqBuilder.add(TermQuery(Term(self.field, term)), BooleanClause.Occur.MUST)
        boolq = boolqBuilder.build()
        hits = searcher.search(boolq, QUERY_MAX_POSTING_LIST).scoreDocs
        return hits

    def _normalized_tf_sampling_and_docFreq(self, term):
        term_ritfs = []
        term_lrtfs = []
        boolqBuilder = BooleanQuery.Builder()
        boolqBuilder.add(TermQuery(Term(self.field, term)), BooleanClause.Occur.MUST)
        boolq = boolqBuilder.build()
        postings = searcher.search(boolq, TERM_MAX_POSTING_LIST).scoreDocs
        docFreq = searcher.count(boolq)
        for post in postings:
            docid = post.doc
            termVec = termVecReader.get(docid, self.field)
            termsEnum = termVec.iterator()
            termsEnum.seekExact(BytesRef(term))
            tf = termsEnum.totalTermFreq()
            dl = termVec.getSumTotalTermFreq()
            num_terms = termVec.size()
            avgtf = dl / num_terms
            avgdl = self.avgdl
            term_ritfs.append(ritf(tf, avgtf))
            term_lrtfs.append(lrtf(tf, avgdl, dl))
        return term_ritfs, term_lrtfs, docFreq

    def EXPTscore(self, docid):
        termVec = termVecReader.get(docid, self.field)
        termsEnum = termVec.iterator()
        dl = termVec.getSumTotalTermFreq()
        num_terms = termVec.size()
        avgtf = dl / num_terms
        avgdl = avgFieldLength(self.collectionStats)
        score = 0
        i = 0
        for term in self.query.split():
            tf_samples = self.tf_sample_list[i]
            lam1 = MLE_lambda_EXP_T(tf_samples[0], self.tau[0])
            lam2 = MLE_lambda_EXP_T(tf_samples[1], self.tau[1])
            termsEnum.seekExact(BytesRef(term))
            tf = termsEnum.totalTermFreq()
            term_ritf = ritf(tf, avgtf)
            term_lrtf = lrtf(tf, avgdl, dl)
            score1 = EXP_T(term_ritf, lam1, self.tau[0])
            score2 = EXP_T(term_lrtf, lam2, self.tau[1])
            idf = iDF(self.docFreq_list[i], self.docCount)
            score += idf * (self.alpha * score1 + (1 - self.alpha) * score2)
            i += 1
        return score

    def scoreDocs(self):
        hits = self._query_interesection_list()
        scoreDocs = []
        for hit in hits:
            docid = hit.doc
            score = self.EXPTscore(docid)
            scoreDoc = (docid, score)
            scoreDocs.append(scoreDoc)
        scoreDocs = sorted(scoreDocs, key=lambda scoreDoc: scoreDoc[1], reverse=True)
        return scoreDocs


def docidTodocno(docid):
    return storedFields.document(docid).get("DOCNO")


query = "turkey iraq water"
field = "TEXT"
tau = (2.737551556053274, 14.527440324502233)  # obtained from running train_trunc_point.py
scorer = EXP_T_Scorer(query, field, tau)

print("-- SCORING --")
for scoredDoc in scorer.scoreDocs():
    print(docidTodocno(scoredDoc[0]), scoredDoc[1])
