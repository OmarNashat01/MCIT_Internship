from flask import Flask, request, jsonify

import Sentiment_analysis_OOP
import pickle

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
NER = spacy.load("en_core_web_lg")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
nltk.download('wordnet')
nltk.download('omw-1.4')

sent_analyzer = pickle.load(open('./model','rb'))


app = Flask(__name__)


@app.route("/", methods=['GET'])
def home():



    args = request.args
    if 'sent' not in args:
        return ('No sentence provided')
    sent = args['sent']

    lem_sent = " ".join([WordNetLemmatizer().lemmatize(word) for word in sent.split()])

    ner_sent = NER(lem_sent)

    dict_dep = { 'Token': [], 'Relation':[],'Head':[],'Children':[]}


    for token in ner_sent:
        dict_dep['Token'].append(str(token.text))
        dict_dep['Relation'].append(str(token.dep_))
        dict_dep['Head'].append(str(token.head.text))
        dict_dep['Children'].append(str([child for child in token.children]))

    response = {
            'feedback' : str(sent_analyzer.predict(sent)[0]),

            'pos tagging': str(nltk.pos_tag(lem_sent.split())),

            'NER' : str([(word.text, word.label_) for word in ner_sent.ents]),

            'DEP': str(dict_dep)
            }

    print(response)
    return (jsonify(response))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3001)
