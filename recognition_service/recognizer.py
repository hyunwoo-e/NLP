from collections import namedtuple
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re
from konlpy.tag import Twitter
import jpype
from threading import Thread

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data


def tokenize(sentence):
    twitter = Twitter()
    for text in sentence:
        tokenized_text = ""
        pos_tagged = twitter.pos(sentence)
        for token in pos_tagged:
            if token[0].isdigit():
                tokenized_text += token[0]+ "/" +'Number' + " "
            if ((token[1] == 'Noun') or (token[1] == 'Verb') or (token[1] == 'Adjective')):
                tokenized_text += token[0] + "/" + token[1] + " "
    return tokenized_text[:tokenized_text.__len__() - 1]


def vectorize(tokenized_train_data):
    corpus = [row[0] for row in tokenized_train_data]
    return corpus


def recognize_service(data):
    jpype.attachThreadToJVM()
    text = str(data["text"])
    preprocessed_text = text

    entityMap = {}

    for key in standard_language_dictionary.keys():
        preprocessed_text = preprocessed_text.replace(key, standard_language_dictionary[key])

    """
    for key in entity_dictionary.keys():
        if (preprocessed_text.find(key)  != -1):
            if entityMap.get(entity_dictionary[key]) == None:
                entityMap[entity_dictionary[key]] = []
            entityMap[entity_dictionary[key]].append(key)
    """

    """
    for key in entity_dictionary.keys():
        for match in re.finditer(key, preprocessed_text):
            if entityMap.get(entity_dictionary[key]) == None:
                entityMap[entity_dictionary[key]] = []
            entityMap[entity_dictionary[key]].append({"value":key, "index":match.start()})
    """

    for key in standard_language_dictionary.keys():
        for match in re.finditer(key, text):
            if entityMap.get(entity_dictionary[standard_language_dictionary[key]]) == None:
                entityMap[entity_dictionary[standard_language_dictionary[key]]] = []
            entityMap[entity_dictionary[standard_language_dictionary[key]]].append({"value":standard_language_dictionary[key], "start":match.start(), "end":match.end()})

    for number in re.findall("\d+", preprocessed_text):
        if entityMap.get("숫자") == None:
            entityMap["숫자"] = []
            entityMap["숫자"].append(number)

    entities = []
    for key in entityMap.keys():
        entities.append({"entity":key, "values":entityMap.get(key)})

    vectorized_text = vect.transform([tokenize(preprocessed_text)]).toarray()

    response = {}
    response["query"] = data["text"]
    response["intent"] = {}
    response["intent"]["intent"] = mlp.predict(vectorized_text)[0]
    response["intent"]["score"] = max(mlp.predict_proba(vectorized_text)[0])
    response["entities"] = entities

    return response



train_data = read_data('train_data.txt')
tokenized_train_data = [(tokenize(row[0]), row[1]) for row in train_data]

vect = TfidfVectorizer()
# vect = CountVectorizer()
vect.fit(vectorize(tokenized_train_data))

TaggedDocument = namedtuple('TaggedDocument', 'words tags')
tokenized_train_docs = [TaggedDocument(d, [c]) for d, c in tokenized_train_data]
train_x = [vect.transform([doc.words]).toarray()[0] for doc in tokenized_train_docs]
train_y = [doc.tags[0] for doc in tokenized_train_docs]

nones = [
    [0, 0, 0] + [0 for i in range(len(train_x[0]) - 3)],
    [0, 0, 1] + [0 for i in range(len(train_x[0]) - 3)],
    [0, 1, 0] + [0 for i in range(len(train_x[0]) - 3)],
    [0, 1, 1] + [0 for i in range(len(train_x[0]) - 3)],
    [1, 0, 0] + [0 for i in range(len(train_x[0]) - 3)],
    [1, 0, 1] + [0 for i in range(len(train_x[0]) - 3)],
    [1, 1, 0] + [0 for i in range(len(train_x[0]) - 3)],
    [1, 1, 1] + [0 for i in range(len(train_x[0]) - 3)]
]

train_x = train_x + nones
train_y = train_y + ["None", "None", "None", "None", "None", "None", "None", "None"]

mlp = MLPClassifier(solver='lbfgs', alpha=0.0025, hidden_layer_sizes=(5, 5), random_state=1234)
mlp.fit(train_x, train_y)

#lreg = LogisticRegression(random_state=1234)
#lreg.fit(train_x, train_y)

#neigh = KNeighborsClassifier(n_neighbors=1)
#neigh.fit(train_x, train_y)

standard_language_data = read_data('standard_language_data.txt')
standard_language_dictionary = {}

for row in standard_language_data:
    for word in row[1].split(','):
        standard_language_dictionary[word] = row[0]

entity_data = read_data('entity_data.txt')
entity_dictionary = {}

for row in entity_data:
    for word in row[1].split(','):
        entity_dictionary[word] = row[0]

#vectorized_text = vect.transform([tokenize("건대입구 123 날짜 시간표")]).toarray()