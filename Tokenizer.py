import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
tmp = stopwords.words('english')

stemmer = PorterStemmer()

file = open("data/stopwords/stop_words_english.txt", 'r', encoding='utf-8')
stopword_list = file.read().split('\n')
file.close()

file = open("data/stopwords/stop_words_english.txt", 'r', encoding='utf-8')
stop_words = file.read().split('\n') 
unfiltered_stopwords = stop_words + tmp

def unfiltered_tokenize(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in unfiltered_stopwords]
    stems = [stemmer.stem(token) for token in tokens]
    return stems

file = open("data/stopwords/useless.txt", 'r', encoding='utf-8')
useless = file.read().split()

L1_stopwords = unfiltered_stopwords + useless
print("Length of new stopwords list:", len(L1_stopwords))

def useless_tokenize(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in L1_stopwords]
    stems = [stemmer.stem(token) for token in tokens]
    return stems