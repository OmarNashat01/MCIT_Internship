from flask import Flask, request, jsonify

from Sentiment_analysis_OOP import ml_model
import pickle


from xgboost import XGBClassifier
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

analyzer = pickle.load(open('model','rb'))

app = Flask(__name__)


@app.route("/", methods=['GET'])
def home():



    args = request.args
    if 'sent' not in args:
        return ('No sentence provided')
    sent = args['sent']

    lem_sent = " ".join([WordNetLemmatizer().lemmatize(word) for word in sent.split()])

    ner_sent = NER(lem_sent)



    dict_dep = {}


    for token in ner_sent:
        dict_dep[str(token.text)] = { 'Relation': str(token.dep_), 'Head': str(token.head.text), 'Children': [child for child in token.children] }

    str(analyzer.predict(sent)[0])
    response = {
            'feedback' : 'Positive' if str(analyzer.predict(sent)[0]) == '1' else 'Negative',

            'pos tagging': str(nltk.pos_tag(lem_sent.split())),

            'NER' : str([(word.text, word.label_) for word in ner_sent.ents]),

            'DEP': str(dict_dep)
            }

    return (jsonify(response))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3001)
