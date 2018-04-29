# https://rare-technologies.com/word2vec-tutorial/
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from database import my_database
import pandas
import nltk
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
import re

def test():
    sentences = [['first', 'sentence'], ['second', 'sentence','king'],['woman','man'],['breakfast', 'cereal', 'dinner', 'lunch','computer']]
    # train word2vec on the two sentences
    model = gensim.models.Word2Vec(sentences, min_count=1)


    # use case
    print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=10))
    #[('queen', 0.50882536)]
    print(model.doesnt_match(['breakfast', 'cereal', 'dinner', 'lunch']))
    #'cereal'
    print(model.similarity('woman', 'man'))
    #0.73723527
    print(model['computer'])  # raw NumPy vector of a word
    #array([-0.00449447, -0.00310097,  0.02421786, ...], dtype=float32)

def main():
    df = my_database.get_data('''(select distinct sum_note from ds_main_ap2_pt);''')
    sentences = []
    for index, row in df.iterrows():
        # Remove all non ASCII characters from unicode string
        text = ''.join([x for x in row['sum_note'] if ord(x) < 128])
        # make lower case
        text = text.lower()
        # tokenise words
        tokens = nltk.word_tokenize(text)
        # stop words  
        stopwords.words('english')  
        clean_tokens = tokens[:] 
        sr = stopwords.words('english')
        
        for index, token in enumerate(clean_tokens):
            #stemming
            clean_tokens[index] = stemmer.stem(token)
            #lemetising
            clean_tokens[index] = lemmatizer.lemmatize(token)
            # remove stop words 
            if token in stopwords.words('english') or len(re.sub('[^a-z]','', token)) < 2:
                if token in clean_tokens:
                    clean_tokens.remove(token)
        sentences += [clean_tokens]
    print (sentences)
    # train word2vec on the two sentences
    model = gensim.models.Word2Vec(sentences, min_count=10)
    model.save('model')
    print(len(sentences))
    #print(model.most_similar(positive=['female'], topn=10))

def match(word):
    model = gensim.models.Word2Vec.load('model')
    print(model.most_similar(positive=word, topn=10))
#main()
match(['shock','sepsis'])