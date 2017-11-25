# -*- coding:utf-8 -*-
import os
import json
import pickle
from random import randint

from pyvi.pyvi import ViTokenizer
from gensim import corpora, matutils
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from flask import Flask, render_template, request, url_for

SPECIAL_CHARACTER = '0123456789%@$.‘​,“”’•…™=+-!;/()*"&^:#|\n\t\''
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

with open(APP_ROOT + "/nb_model.pkl", 'rb') as file:
    estimator_nb = pickle.load(file)

class FileStore(object):
    def __init__(self, filePath, data = None):
        self.filePath = filePath
        self.data = data

    def store_json(self):
        with open(self.filePath, 'w') as outfile:
            json.dump(self.data, outfile, ensure_ascii=False)

    def store_dictionary(self, dict_words):
        dictionary = corpora.Dictionary(dict_words)
        dictionary.filter_extremes(no_below=20, no_above=0.3)
        dictionary.save_as_text(self.filePath)

    def save_pickle(self,  obj):
        with open(self.filePath, 'wb') as file:
            pickle.dump(obj, file)

class FileReader(object):
    def __init__(self, filePath):
        self.filePath = filePath

    def read(self):
        with open(self.filePath) as f:
            s = f.read()
        return s

    def content(self):
        s = self.read()
        return s

    def read_json(self):
        with open(self.filePath) as f:
            s = json.loads(f.read().decode("utf-8", errors="ignore"))
        return s

    def read_stopwords(self):
        with open(self.filePath, 'r') as f:
            stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])
        return stopwords

    def load_dictionary(self):
        return corpora.Dictionary.load_from_text(self.filePath)

class NLP(object):
    def __init__(self, text = None):
        self.text = text
        self.__set_stopwords()

    # Lấy stopwords từ file stopwords-vi.txt
    def __set_stopwords(self):
        self.stopwords = FileReader(APP_ROOT + "/stopwords-vi.txt").read_stopwords()

    # Tách từ sử dụng thư viện ViTokenizer
    def segmentation(self):
        return ViTokenizer.tokenize(self.text)

    # Loại bỏ các ký tự đặc biệt
    def split_words(self):
        text = self.segmentation()
        try:
            return [x.strip(SPECIAL_CHARACTER.decode("utf-8")).lower() for x in text.split()]
        except TypeError:
            return []

    # Loại bỏ stop words, Lấy feature words
    def get_words_feature(self):
        split_words = self.split_words()
        return [word for word in split_words if word.encode('utf-8') not in self.stopwords]

class FeatureExtraction(object):
    def __init__(self, data):
        self.data = data

    def build_dictionary(self):
        print 'Building dictionary'
        dict_words = []
        i = 0
        for text in self.data:
            i += 1
            print "Build Dict Step {} / {}".format(i, len(self.data))
            words = NLP(text=text['content']).get_words_feature()
            dict_words.append(words)
        FileStore(filePath= APP_ROOT + "/dictionary.txt").store_dictionary(dict_words)

    def load_dictionary(self):
        if os.path.exists(APP_ROOT + "/dictionary.txt") == False:
            self.build_dictionary()
        self.dictionary = FileReader(APP_ROOT+ "/dictionary.txt").load_dictionary()

    def get_dense(self, text):
        self.load_dictionary()
        words = NLP(text).get_words_feature()
        # Bag of words
        vec = self.dictionary.doc2bow(words)
        dense = list(matutils.corpus2dense([vec], num_terms=len(self.dictionary)).T[0])
        return dense

    def __build_dataset(self):
        self.features = []
        self.labels = []
        i = 0
        for d in self.data:
            i += 1
            print "Build Dataset Step {} / {}".format(i, len(self.data))
            self.features.append(self.get_dense(d['content']))
            self.labels.append(d['category'])

    def get_data_and_label(self):
        self.__build_dataset()
        return self.features, self.labels

def _get_article(key):
    if key == "giai_tri":
        return "Giải trí".decode("utf-8")
    elif key == "giao_duc_khuyen_hoc":
        return "Giáo dục".decode("utf-8")
    elif key == "kinh_doanh":
        return "Kinh doanh".decode("utf-8")
    elif key == "o_to_xe_may":
        return "Xe".decode("utf-8")
    elif key == "phap_luat":
        return "Pháp luật".decode("utf-8")
    elif key == "suc_khoe":
        return "Sức khỏe".decode("utf-8")
    elif key == "suc_manh_so":
        return "Công nghệ".decode("utf-8")
    elif key == "the_gioi":
        return "Thế giới".decode("utf-8")
    elif key == "tinh_yeu_gioi_tinh":
        return "Tình yêu & Giới tính".decode("utf-8")
    elif key == "van_hoa":
        return "Văn hóa".decode("utf-8")
    elif key == "xa_hoi":
        return "Xã hội".decode("utf-8")
    elif key == "the_thao":
        return "Thể thao".decode("utf-8")
    else:
        return "Unkown".decode("utf-8")

@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        feature_extraction = FeatureExtraction(data="").get_dense(request.form["news"])
        article = estimator_nb.predict([feature_extraction])
        return render_template("index.html", article = _get_article(article[0].decode("utf-8")))
    else:
        return render_template("index.html")


if __name__ == '__main__':
    app.run()