import json
import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("en")  # replace "en" with your model's language if different
doc_bin = DocBin()


with open("cats.jsonl", "r") as file:
    lines = file.readlines()
    i = 0
    for line in lines:
        if i > 2500:
            item = json.loads(line)
            text = item.get('text', '')
            cats = item.get('categories', {})
            doc = nlp.make_doc(text)
            doc.cats = cats
            doc_bin.add(doc)
        i += 1

doc_bin.to_disk("./dev.spacy")