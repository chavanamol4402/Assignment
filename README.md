# Installation
pip install -U assemblyai
pip install spacy
python -m spacy download en_core_web_sm

# Import assemblyai and use your API key     

import assemblyai as aai

aai.settings.api_key = "use your API key "
# transciber instance
transcriber = aai.Transcriber()
# Audio file Input to transciber

transcript = transcriber.transcribe("/content/sales_call_telephone_marketers.wav")
print(transcript.text)
# transcibed output     
Hello? Hi, Nancy. This is Mike from at and T, Inc. Yes, how can I help you? Nancy, you have been using our prepaid connection for a couple of years now, right? Yeah, that's right. How would you like a postpaid connection that allows you to make free, unlimited voice calls to three at and T numbers? I would love that, but what's the catch? There's no catch. There will be a monthly rental which you will have to pay like any other postpaid connection. Fantastic. Sign me up.

trans_text= transcript.text
     
# import spacy for NLP task
import spacy
import random
from spacy.training import Example

# customised training data
TRAINING_DATA = [
    ("This is a call from Apple, and my name is Tim Cook.", {"entities": [(20, 26, "ORG"), (42, 51, "caller_name")]}),
    ("I am calling on behalf of IBM, and I am John Smith.", {"entities": [(26, 30, "ORG"), (40, 51, "caller_name")]}),
    ("Hello, this is Mark Zuckerberg calling from Facebook.", {"entities": [(15, 31, "caller_name"), (44, 53, "ORG")]}),
    ("I am calling about your Amazon Prime membership.", {"entities": [(23, 48, "PRODUCT")]}),
    ("This call is regarding your Netflix subscription.",  {"entities": [(27, 49, "PRODUCT")]}),
    ("I am Lisa, calling from Tesla Motors.", {"entities": [(5, 10, "caller_name"), (24, 37, "ORG")]}),
    ("This is Mike from at and T, Inc.", {"entities": [(18, 32, "ORG"), (8, 13, "caller_name")]}),
    ("I am calling on behalf of T-Mobile, and my name is Sarah.", {"entities": [(26, 35, "ORG"), (51, 57, "caller_name")]}),
    ("This call is regarding your prepaid connection.",  {"entities": [(28, 47, "PRODUCT")]}),
    ("This call is regarding your postpaid connection.",  {"entities": [(28, 48, "PRODUCT")]}),
    ("This call is regarding your Microsoft Office subscription.", {"entities": [(28, 58, "PRODUCT")]}),
    ("I am John, calling from Oracle Corporation.", {"entities": [(5, 10, "caller_name"), (24, 43, "ORG")]}),
    ("This is a call from Cisco Systems, and I am Alex.", {"entities": [(24, 34, "ORG"), (44, 49, "caller_name")]}),
    ("I am calling on behalf of at and T, Inc, and my name is Mike.", {"entities": [(26, 40, "ORG"), (56, 61, "caller_name")]}),
    ("Hello, this is Larry Page calling from Google.", {"entities": [(15, 26, "caller_name"), (39, 46, "ORG")]}),
    ("Hello, this is Nancy Page calling from at & T.", {"entities": [(15, 21, "caller_name"), (39, 46, "ORG")]}),
    
]

     
# Entity lable list
LABELS = ["caller_name", "ORG", "PRODUCT"]
     
# Instance for NER model
def train_ner_model(training_data, labels, iterations=30):
    nlp = spacy.blank("en")

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    for label in labels:
        ner.add_label(label)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.initialize()

        for _ in range(iterations):
            random.shuffle(training_data)
            losses = {}
            for text, annotations in training_data:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)

    output_dir = "/content/trained_ner_model"  # Change this path as needed
    nlp.to_disk(output_dir)
     
# Train NER model on Training data
train_ner_model(TRAINING_DATA, LABELS)

# Load Trained NER Model
nlp = spacy.load("trained_ner_model")
     
# Feed transcribed text ro NER model
doc = nlp(trans_text)
     
# Get Entities from transcribed text
for ent in doc.ents:
    print(ent.text, ent.label_)
# Entity output    
Nancy. ORG
at and T, Inc. ORG
Yes, caller_name
Nancy, ORG
prepaid connection for PRODUCT
postpaid connection that PRODUCT
at and T numbers? I ORG
There's ORG
There will ORG
pay like caller_name
postpaid connection. PRODUCT
Fantastic. caller_name
up. caller_name

# split transcribed text to list of sentences
import re
sentences = re.split(r'(?<=[.?!])\s+', trans_text)
sentences
     
['Hello?',
 'Hi, Nancy.',
 'This is Mike from at and T, Inc.',
 'Yes, how can I help you?',
 'Nancy, you have been using our prepaid connection for a couple of years now, right?',
 "Yeah, that's right.",
 'How would you like a postpaid connection that allows you to make free, unlimited voice calls to three at and T numbers?',
 "I would love that, but what's the catch?",
 "There's no catch.",
 'There will be a monthly rental which you will have to pay like any other postpaid connection.',
 'Fantastic.',
 'Sign me up.']
 
# Dump the output to JSON file.
task_3_output = []
for sentence in sentences:
    doc = nlp(sentence)
    entities = []
    for ent in doc.ents:
        entities.append({
            "entity_name": ent.label_,
            "entity_value": ent.text
        })
    task_3_output.append({
        "sentence": sentence,
        "entities": entities
    })

output = {
    "task_1_output": trans_text,
    "task_3_output": task_3_output
}
     

import json
with open("output.json", "w") as outfile:
    json.dump(output, outfile)

# read JSON file     

file_path = "output.json"

with open(file_path, "r") as file:
    json_data = json.load(file)

print(json_data)
     
{'task_1_output': "Hello? Hi, Nancy. This is Mike from at and T, Inc. Yes, how can I help you? Nancy, you have been using our prepaid connection for a couple of years now, right? Yeah, that's right. How would you like a postpaid connection that allows you to make free, unlimited voice calls to three at and T numbers? I would love that, but what's the catch? There's no catch. There will be a monthly rental which you will have to pay like any other postpaid connection. Fantastic. Sign me up.", 'task_3_output': [{'sentence': 'Hello?', 'entities': []}, {'sentence': 'Hi, Nancy.', 'entities': [{'entity_name': 'ORG', 'entity_value': 'Nancy.'}]}, {'sentence': 'This is Mike from at and T, Inc.', 'entities': [{'entity_name': 'ORG', 'entity_value': 'at and T, Inc.'}]}, {'sentence': 'Yes, how can I help you?', 'entities': [{'entity_name': 'caller_name', 'entity_value': 'Yes,'}]}, {'sentence': 'Nancy, you have been using our prepaid connection for a couple of years now, right?', 'entities': [{'entity_name': 'ORG', 'entity_value': 'Nancy,'}, {'entity_name': 'PRODUCT', 'entity_value': 'prepaid connection for'}]}, {'sentence': "Yeah, that's right.", 'entities': []}, {'sentence': 'How would you like a postpaid connection that allows you to make free, unlimited voice calls to three at and T numbers?', 'entities': [{'entity_name': 'PRODUCT', 'entity_value': 'postpaid connection that'}, {'entity_name': 'ORG', 'entity_value': 'at and T numbers?'}]}, {'sentence': "I would love that, but what's the catch?", 'entities': []}, {'sentence': "There's no catch.", 'entities': [{'entity_name': 'ORG', 'entity_value': "There's"}]}, {'sentence': 'There will be a monthly rental which you will have to pay like any other postpaid connection.', 'entities': [{'entity_name': 'ORG', 'entity_value': 'There will'}, {'entity_name': 'caller_name', 'entity_value': 'pay like'}, {'entity_name': 'PRODUCT', 'entity_value': 'postpaid connection.'}]}, {'sentence': 'Fantastic.', 'entities': [{'entity_name': 'caller_name', 'entity_value': 'Fantastic.'}]}, {'sentence': 'Sign me up.', 'entities': []}]}
