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
from nltk.corpus import stopwords
import re
#nltk.download()
relevant_feature =  [
                        'shock',
                        'septic',
                        'bp',
                        'er',
                        'load',
                        'line',
                        'nss',
                        'levophed',
                        'drop',
                        'central',
                        'heart',
                        'neutropenia',
                        'imp',
                        #'ext',
                        'aki',
                        'febrile',
                        'abd',
                        #'pi',
                        'cvp',
                        'hydrocortisone',
                        'murmur',
                        'nephro',
                        #'ml',
                        'acidosis',
                        #'cc',
                        'regular',
                        #'heent',
                        'rt',
                        'hemodynamic',
                        'pale',
                        'metabolic',
                        'ward',
                        'pr',
                        'empirical',
                        'atn',
                        'drip',
                        'soft',
                        'lungs',
                        'rr',
                        'bun/cr',
                        'hypovolemic',
                        'access',
                        'edema',
                        'crrt',
                        'u/d',
                        'meropenem',
                        'secretion',
                        'anc',
                        'jx',
                        'cast',
                        #'lab',
                        'cpr',
                        'bma',
                        'sound',
                        'ua',
                        'blood',
                        'sputum',
                        'serum',
                        'yr',
                        '=-ve',
                        's1s2',
                        'gross',
                        'creatinine',
                        'stable',
                        'med',
                        'sec',
                        'fine',
                        'g/s',
                        'muddy',
                        'lavage',
                        'na',
                        'h/d',
                        'baseline',
                        'ft3',
                        'ugih',
                        'consult',
                        'iii',
                        'iv',
                        'segment',
                        'set',
                        'lpm',
                        'wbc=',
                        'chemo',
                        'tracheostomy',
                        'cirrhosis',
                        'aml',
                        'ft4',
                        'male',
                        'advice',
                        'cardiogenic',
                        'subicu',
                        'airway',
                        'echo',
                        'start',
                        'lt.',
                        'ceftazidime',
                        'cloxacillin',
                        'tft',
                        'sub',
                        'hco3'
                    ]
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
def get_value(tokens,index):
    if index+2 < len(tokens):
        v = tokens[index+1:index+2][0]
        #print(tokens[index-1:index+2])

        if len(v.split('=')) > 1:
            return re.sub(r'[^\d.]+','', v.split('=')[1])
        if len(v.split('-')) > 1:
            return re.sub(r'[^\d.]+','', v.split('-')[0])
        if len(v.split('/')) > 1:
            return re.sub(r'[^\d.]+','', v.split('/')[0])
        return re.sub(r'[^\d.]+','', v)
    else:
        return ''

def find_min(v):
    if len(v) > 0:
        return min(v)
    else:
        return ''
def main():
    df = my_database.get_data('''(select sum_note,dx1_nm, dx2_nm, dx3_nm 
                                 from ds_main_ap2_pt 
                                 WHERE CONCAT(dx1_nm, dx2_nm, dx3_nm) like '%septicaemia%' 
                                 limit 5000)
                                 UNION
                                 (select sum_note,dx1_nm, dx2_nm, dx3_nm 
                                 from ds_main_ap2_pt 
                                 WHERE CONCAT(dx1_nm, dx2_nm, dx3_nm) like '%shock%' 
                                 limit 4000);''')
    caseset = []
    cols = ['bp_value','rr_value','temp_value','pao2_value','plt_value','wbc_value','cr_value']

    for index, row in df.iterrows():
        target = '1_no_shock'
        # if sepsis with shock
        if any("shock" in s for s in [row['dx1_nm'].lower(), row['dx2_nm'].lower(), row['dx3_nm'].lower()]):
            target = '0_shock'
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
        stopwords.words('english')  
        clean_tokens = tokens[:] 
        sr = stopwords.words('english')
        bp = []
        rr = []
        temp = []
        pao2 = []
        plt = []
        wbc = []
        cr = []
        for index, token in enumerate(tokens):
            #print(token)
            tokens[index] = stemmer.stem(token)
            tokens[index] = lemmatizer.lemmatize(token)

            if 'bp' in tokens[index]:
                bp += [get_value(tokens,index)]
            if 'rr' == tokens[index]:
                rr += [get_value(tokens,index)]
            if 'temp' in tokens[index]:
                temp += [get_value(tokens,index)]
            if 'pao2' in tokens[index]:
                pao2 += [get_value(tokens,index)]
            if 'plt' in tokens[index]:
                plt += [get_value(tokens,index)]
            if 'wbc' in tokens[index]:
                wbc += [get_value(tokens,index)]
            if 'cr' == tokens[index]:
                cr += [get_value(tokens,index)]
                
            if token in stopwords.words('english') or len(re.sub('[^a-z]','', token)) < 2 or token == 'shock' or token not in relevant_feature:
                clean_tokens.remove(token)
                
        value = [find_min(bp),find_min(rr),find_min(temp),find_min(pao2),find_min(plt),find_min(wbc),find_min(cr)]
        value = [word.replace('.','') for word in value]
        # count word frequency
        freq = nltk.FreqDist(clean_tokens) 
        v = list(set(clean_tokens))

        caseset += [pandas.DataFrame(data=[(v+value+[target])], columns=(v+cols+['target_class']))]
        #print(freq.items())
        #freq.plot(20, cumulative=False)
        #for key,val in freq.items(): 
        #    print (str(key) + ':' + str(val))


    result = caseset[0].append(caseset[1:len(caseset)])

    col_list = result.columns.tolist()

    for col in cols:
        col_list.remove(col)
    col_list.remove('target_class')
    #sort column
    col_list.sort()
    #replace value with 0/1
    feature = (result[col_list].notnull()).astype('int')
    #remove column contains < 5 values
    feature.drop([col for col, val in feature.sum().iteritems() if val < 5], axis=1, inplace=True)
    target_class = result[cols+['target_class']]
    dataset = pandas.concat([feature,target_class], axis=1)
    dataset.drop_duplicates(inplace=True)
    dataset.to_csv('result1.csv', index=False)
    
main()

