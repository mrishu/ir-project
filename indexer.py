#!/usr/bin/env python3

""" -- INDEXING --

There are four types of files, starting with:
fb, fr, ft, la.
Each file contains lots of documents.

Each document is inside a <doc> tag which contains many
children tags which enclose different fields for each document.

We are indexing two fields from each document, namely: DOCNO, TEXT.
1. DOCNO is inside <docno> tag in all four types of files.
2. TEXT  is inside <text>  tag in all four types of files.

- We are storing DOCNO as StringField (as it is not tokenized).
- We are NOT storing TEXT but indexing it as TextField (as it is tokenized),
    and we are also storing the TermVectors for this field.

NOTE:
- <docno> tag is present in ALL documents, but some dont contain <text> tag.
- We ignore documents with no TEXT i.e. no <text> tag.
- We ignore information inside all other tags for all files.
"""


import os
from bs4 import BeautifulSoup

import lucene
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import FSDirectory
import org.apache.lucene.document as document

lucene.initVM()

# Make sure index-dir directory is removed before re-indexing
indexPath = File("index-dir/").toPath()
indexDir = FSDirectory.open(indexPath)
analyzer = StandardAnalyzer()
writerConfig = IndexWriterConfig(analyzer)
writer = IndexWriter(indexDir, writerConfig)


# Make new IndexableFieldType that stores term vectors (we are doing this for TEXT field)
termvecstore_TextField = document.FieldType(document.TextField.TYPE_NOT_STORED)
termvecstore_TextField.setStoreTermVectors(True)


def indexDoc(docno, text):
    doc = document.Document()
    doc.add(document.Field("DOCNO", docno, document.StringField.TYPE_STORED))
    doc.add(document.Field("TEXT", text, termvecstore_TextField))
    writer.addDocument(doc)


doc_count = 1
for filename in os.listdir("./documents"):
    with open("./documents/" + filename, "r", encoding="ISO-8859-1") as fp:
        soup = BeautifulSoup(fp, "html.parser")
        doc = soup.find("doc")
        while doc is not None:
            docno = doc.findChildren("docno")[0].get_text().strip()
            text = doc.findChildren("text")
            # ignore document if no <text> tag present
            if len(text) == 0:
                doc = doc.find_next("doc")
                continue
            text = text[0].get_text().strip()
            # it can be that <text> tag is present but nothing is inside. ignore that too.
            if text == "":
                doc = doc.find_next("doc")
                continue
            print(f"Indexing {doc_count} -- {docno}")
            indexDoc(docno, text)
            doc = doc.find_next("doc")
            doc_count += 1

writer.close()
