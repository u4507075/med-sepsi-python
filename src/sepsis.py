# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

from database import my_database
import pandas
import nltk
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


#nltk.download()

df = my_database.get_data("select sum_note,dx1_nm, dx2_nm, dx3_nm from ds_main_ap2_pt WHERE CONCAT(dx1_nm, dx2_nm, dx3_nm) like '%septicaemia%' or CONCAT(dx1_nm, dx2_nm, dx3_nm) like '%septic shock%' limit 2;")
caseset = []
for index, row in df.iterrows():
    target = 'no_shock'
    if any("shock" in s for s in [row['dx1_nm'].lower(), row['dx2_nm'].lower(), row['dx3_nm'].lower()]):
        target = 'shock'
    # Remove all non ASCII characters from unicode string
    text = ''.join([x for x in row['sum_note'] if ord(x) < 128])
    # make lower case
    text = text.lower()
    #print(text)
    tokens = nltk.word_tokenize(text)
    print(tokens)
    two_words_tokens = []
    for index, token in enumerate(tokens):
        if index < len(tokens)-1:
            two_words_tokens += [token+' '+tokens[index+1]] 
        else:
            two_words_tokens += [token] 
    #print(two_words_tokens)
    
    #tokens = two_words_tokens
    # remove stop words
    from nltk.corpus import stopwords
    stopwords.words('english')  
    clean_tokens = tokens[:] 
    sr = stopwords.words('english')
    for index, token in enumerate(tokens):
        #print(token)
        tokens[index] = stemmer.stem(token)
        tokens[index] = lemmatizer.lemmatize(token)
        if token in stopwords.words('english'):
            clean_tokens.remove(token)

    # count word frequency
    freq = nltk.FreqDist(clean_tokens) 
    v = list(set(clean_tokens))

    caseset += [pandas.DataFrame(data=[(v+[target])], columns=(v+['target_class']))]
    #print(freq.items())
    #freq.plot(20, cumulative=False)
    #for key,val in freq.items(): 
    #    print (str(key) + ':' + str(val))

result = caseset[0].append(caseset[1:len(caseset)])
col_list = result.columns.tolist()
col_list.remove('target_class')
col_list += ['target_class']
result = result[col_list]
print(result)
result.to_csv('result.csv', index=False)