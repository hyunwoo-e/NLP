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
            #print(tokenized_text)
    return tokenized_text[:tokenized_text.__len__() - 1]


def vectorize(tokenized_train_data):
    corpus = [row[0] for row in tokenized_train_data]
    return corpus


def recognize_service(data):
    jpype.attachThreadToJVM()
    text = str(data["text"])
    preprocessed_text = text + " "
    entityMap = {}

    #print(preprocessed_text)
    for key in standard_language_dictionary.keys():
        preprocessed_text = preprocessed_text.replace(key, standard_language_dictionary[key])
    #print(preprocessed_text)

    for key in entity_dictionary.keys():
        for match in re.finditer(key, preprocessed_text):
            if entityMap.get(entity_dictionary[key]) == None:
                entityMap[entity_dictionary[key]] = []
            entityMap[entity_dictionary[key]].append({"value":key, "start":match.start(), "end":match.end()})

    for key in entity_dictionary.keys():
        preprocessed_text = preprocessed_text.replace(key, entity_dictionary[key])

    for match in re.finditer("\d{3,4}-\d{3,4}-\d{4}", preprocessed_text):
        if entityMap.get("전화번호") == None:
            entityMap["전화번호"] = []
        entityMap["전화번호"].append({"value": match.group(), "start": match.start(), "end": match.end()})
        preprocessed_text = preprocessed_text.replace(match.group(), "전화번호", 1)

    for match in re.finditer("(\d{4})-(\d{1,2})-(\d{1,2})", preprocessed_text):
        if entityMap.get("날짜") == None:
            entityMap["날짜"] = []
        entityMap["날짜"].append({"value": match.group(1)+"년", "start": match.start(1), "end": match.end(1)})
        entityMap["날짜"].append({"value": match.group(2)+"월", "start": match.start(2), "end": match.end(2)})
        entityMap["날짜"].append({"value": match.group(3)+"일", "start": match.start(3), "end": match.end(3)})
        preprocessed_text = preprocessed_text.replace(match.group(), "날짜", 1)

    for match in re.finditer("(\d{4}년)[ ]*(\d{1,2}월)[ ]*(\d{1,2}일)", preprocessed_text):
        if entityMap.get("날짜") == None:
            entityMap["날짜"] = []
        entityMap["날짜"].append({"value": match.group(1), "start": match.start(1), "end": match.end(1)})
        entityMap["날짜"].append({"value": match.group(2), "start": match.start(2), "end": match.end(2)})
        entityMap["날짜"].append({"value": match.group(3), "start": match.start(3), "end": match.end(3)})
        preprocessed_text = preprocessed_text.replace(match.group(), "날짜", 1)

    for match in re.finditer("(\d{1,2}월)[ ]*(\d{1,2}일)", preprocessed_text):
        if entityMap.get("날짜") == None:
            entityMap["날짜"] = []
        entityMap["날짜"].append({"value": match.group(1), "start": match.start(1), "end": match.end(1)})
        entityMap["날짜"].append({"value": match.group(2), "start": match.start(2), "end": match.end(2)})
        entityMap["날짜"].append({"value": match.group(3), "start": match.start(3), "end": match.end(3)})
        preprocessed_text = preprocessed_text.replace(match.group(), "날짜", 1)

    for match in re.finditer("(\d{4}년)[ ]*(\d{1,2}월)[ ]*", preprocessed_text):
        if entityMap.get("날짜") == None:
            entityMap["날짜"] = []
        entityMap["날짜"].append({"value": match.group(1), "start": match.start(1), "end": match.end(1)})
        entityMap["날짜"].append({"value": match.group(2), "start": match.start(2), "end": match.end(2)})
        entityMap["날짜"].append({"value": match.group(3), "start": match.start(3), "end": match.end(3)})
        preprocessed_text = preprocessed_text.replace(match.group(), "날짜", 1)

    for match in re.finditer("\d{4}년", preprocessed_text):
        if entityMap.get("날짜") == None:
            entityMap["날짜"] = []
        entityMap["날짜"].append({"value": match.group()[:len(match.group())-1]+"년", "start": match.start(), "end": match.end()})
        preprocessed_text = preprocessed_text.replace(match.group(), "날짜", 1)

    for match in re.finditer("\d{1,2}월", preprocessed_text):
        if entityMap.get("날짜") == None:
            entityMap["날짜"] = []
        entityMap["날짜"].append({"value": match.group()[:len(match.group())-1]+"월", "start": match.start(), "end": match.end()})
        preprocessed_text = preprocessed_text.replace(match.group(), "날짜", 1)

    for match in re.finditer("\d{1,2}일", preprocessed_text):
        if entityMap.get("날짜") == None:
            entityMap["날짜"] = []
        entityMap["날짜"].append({"value": match.group()[:len(match.group())-1]+"일", "start": match.start(), "end": match.end()})
        preprocessed_text = preprocessed_text.replace(match.group(), "날짜", 1)

    for match in re.finditer("\d+", preprocessed_text):
        if entityMap.get("숫자") == None:
            entityMap["숫자"] = []
        entityMap["숫자"].append({"value": match.group(), "start": match.start(), "end": match.end()})
        preprocessed_text = preprocessed_text.replace(match.group(), "숫자", 1)

    for key in entity_dictionary.keys():
        for match in re.finditer(key+"[가-힣]*"+"[에서|출발] ", preprocessed_text):
            if entityMap.get("출발지") == None:
                entityMap["출발지"] = []
            entityMap["출발지"].append({"value": key, "start": match.start(), "end": match.end()})

    for key in entity_dictionary.keys():
        for match in re.finditer(key+"[가-힣]*"+"[에|로|으로|가는|행|도착] ", preprocessed_text):
            if entityMap.get("목적지") == None:
                entityMap["목적지"] = []
            entityMap["목적지"].append({"value": key, "start": match.start(), "end": match.end()})
    #print(preprocessed_text)

    entities = []
    for key in entityMap.keys():
        entities.append({"entity":key, "values":entityMap.get(key)})

    entity_labeled_text = preprocessed_text
    '''
    entity_labeled_text = preprocessed_text
    print(entity_labeled_text)
    for key in entity_dictionary.keys():
        entity_labeled_text = entity_labeled_text.replace(key, entity_dictionary[key])
    print(entity_labeled_text)
    '''

    print(entity_labeled_text)
    print(tokenize(entity_labeled_text))
    vectorized_text = vect.transform([tokenize(entity_labeled_text)]).toarray()
    print(vectorized_text)

    print(mlp.predict(vectorized_text)[0])
    print(lreg.predict(vectorized_text)[0])
    print(neigh.predict(vectorized_text)[0])

    response = {}
    response["query"] = text
    response["preprocessed_query"] = preprocessed_text
    response["intent"] = {}
    response["intent"]["intent"] = mlp.predict(vectorized_text)[0]
    response["intent"]["score"] = max(mlp.predict_proba(vectorized_text)[0])
    response["entities"] = entities

    print(response)
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

mlp = MLPClassifier(solver='lbfgs', alpha=0.002, hidden_layer_sizes=(10, 10), random_state=1234)
mlp.fit(train_x, train_y)

mlp_train_x = train_x.copy()
mlp_train_x.append(train_x[0])
mlp_train_x.append(train_x[0])
mlp_train_x.append(train_x[0])
mlp_train_x.append(train_x[0])
mlp_train_x.append(train_x[0])
mlp_train_y = train_y.copy()
mlp_train_y.append(train_y[0])
mlp_train_y.append(train_y[0])
mlp_train_y.append(train_y[0])
mlp_train_y.append(train_y[0])
mlp_train_y.append(train_y[0])
mlp.fit(mlp_train_x, mlp_train_y)

lreg = LogisticRegression(random_state=1234)
lreg.fit(train_x, train_y)

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(train_x, train_y)

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