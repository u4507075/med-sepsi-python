import pandas
import re
from collections import Counter

path = '../data/sepsis/'

#Step 1: create a training set
'''
sum_note = ['sum_note']
dx = []

df = pandas.concat([pandas.read_csv(path+'ds_main_ap_pt.csv'),pandas.read_csv(path+'ds_main_ap2_pt.csv')])
#df.drop_duplicates(inplace=True)
df['sum_note'] = df['sum_note'].apply(lambda x: str(x).lower())
df['sum_note'] = df['sum_note'].apply(lambda x: re.sub('[^a-z]', ' ', x).split(' '))
df['sum_note'] = df['sum_note'].apply(lambda x: list(filter(None, x)))

for i in range(1,14):
	i = 'dx'+str(i)+'_nm'
	dx.append(i)
	df[i] = df[i].apply(lambda x: str(x).lower())
	df[i] = df[i].apply(lambda x: re.sub('[^a-z]', ' ', x))

df['dx_all'] = df['dx1_nm']+df['dx2_nm']+df['dx3_nm']+df['dx4_nm']+df['dx5_nm']+df['dx6_nm']+df['dx7_nm']+df['dx8_nm']+df['dx9_nm']+df['dx10_nm']+df['dx11_nm']+df['dx12_nm']+df['dx13_nm']
df['sepsis'] = df['dx_all'].apply(lambda x: True if ('sepsis' in x or 'septicemia' in x or 'septicaemia' in x) else False)
df = df[sum_note+['sepsis']]
df.to_csv(path+'trainingset.csv')
'''
#Step 2: select relevant features

from sklearn import preprocessing
from sklearn import model_selection
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import ast
from collections import Counter
def get_dataset(trainingset, validation_size):
	n = len(trainingset.columns)-1
	array = trainingset.values
	X = array[:,0:n]
	Y = array[:,n]
	seed = 7
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
	X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
	return X_train, X_validation, Y_train, Y_validation

c = Counter()
df = pandas.read_csv(path+'trainingset.csv')
for i in range(1,1000):
	d = df.sample(n=1000)
	d['sum_note'] = d['sum_note'].apply(lambda x: ast.literal_eval(x))
	d['sepsis'] = d['sepsis'].apply(lambda x: str(x))
	l = d[['sepsis']]
	d = d[['sum_note']]
	from sklearn.preprocessing import MultiLabelBinarizer
	mlb = MultiLabelBinarizer()
	d = d.join(pandas.DataFrame(mlb.fit_transform(d.pop('sum_note')),columns=mlb.classes_,index=d.index))
	r = pandas.concat([d, l], axis=1, join_axes=[d.index])

	X_train, X_validation, Y_train, Y_validation = get_dataset(r, 0.2)
	lsvc = LinearSVC(C=0.01, penalty="l1", dual=False, class_weight='balanced')
	model = SelectFromModel(lsvc).fit(X_train, Y_train)
	print(accuracy_score(Y_validation, lsvc.fit(X_train, Y_train).predict(X_validation)))
	c.update(d.columns[model.get_support(indices=True)].tolist())
	print(c)

'''
features = []
for i in range(2011,2018):
	df = pandas.read_excel(path+str(i)+'.xls')
	f = pandas.read_csv('feature.csv')
	df['pre_operation'] = df['pre_operation'].apply(lambda x: str(x).lower())
	df['pre_operation'] = df['pre_operation'].apply(lambda x: re.sub('[^a-z]', ' ', x).split(' '))
	df['pre_operation'] = df['pre_operation'].apply(lambda x: list(filter(None, x)))
	feature = f.columns.tolist()
	s = df.pre_operation.apply(lambda x: pandas.Series(x)).unstack()
	df2 = df.join(pandas.DataFrame((s.reset_index(level=0, drop=True)))).rename(columns={0:'feature'})
	df2 = df2[['feature','icd9']]
	df2 = df2[df2.feature.notnull()]
	df2 = df2[df2.icd9.notnull()]
	df2['icd9'] = df2['icd9'].apply(lambda x: '#'+str(x))
	features.append(df2)
	print(i)
df = pandas.concat(features)
df.to_csv('trainingset.csv')
'''

#Step 3: create a word2vec model capturing association between pre-operation words
'''
df = pandas.read_csv('trainingset.csv')
df = df[['feature','icd9']]
df = df[df.feature.notnull()]
import gensim
model = gensim.models.Word2Vec(df.values.tolist(), min_count=1)
model.save('model')
'''

#Step 4: use the word2vec model to find the associated words and link them back to icd-9
'''
import gensim
df_t = pandas.read_csv('trainingset.csv')
model = gensim.models.Word2Vec.load('model')
f = pandas.read_csv('feature.csv')
feature = f.columns.tolist()
dftest = pandas.read_excel(path+'2018.xls')
dftest['pre_operation_list'] = dftest['pre_operation'].apply(lambda x: str(x).lower())
dftest['pre_operation_list'] = dftest['pre_operation_list'].apply(lambda x: re.sub('[^a-z]', ' ', x).split(' '))
dftest['pre_operation_list'] = dftest['pre_operation_list'].apply(lambda x: list(filter(None, x)))
#dftest['recommendation_level'] = 0
#dftest['probability'] = ''
#dftest['recommended_icd9'] = ''
#dftest['recommended_icd9_probability'] = ''
#dftest = dftest.head(20)
for index,row in dftest.iterrows():
	if 'nan' not in row['pre_operation_list']:
		v = [x for x in row['pre_operation_list'] if x in f]
		if len(v) > 0:
			similar_words = model.most_similar(positive=v, topn=10)
			words = []
			for i in similar_words:
				words.append(i[0])
			df = df_t.copy()
			df = df[df.feature.isin(words)]
			df = pandas.DataFrame(df['icd9'].value_counts(normalize=True)).reset_index()
			df = df.rename(columns={'index':'icd9','icd9':'probability'})
			df['icd9'].replace({'#': ''}, inplace=True, regex=True)
			#dftest.at[index,'recommended_icd9'] = str(df.head(5)['icd9'].tolist())
			#dftest.at[index,'recommended_icd9_probability'] = str(df.head(5)['probability'].tolist())
			result = df[df['icd9'] == str(row['icd9'])]
			
			if len(result) == 1:
				dftest.at[index,'recommendation_level'] = result.index.values[0]+1
				dftest.at[index,'probability'] = result['probability'].values.tolist()[0]
			
				
print(dftest)
dftest.replace({';': ''}, inplace=True, regex=True)
dftest = dftest[['icd9','descs','pre_operation','recommendation_level']]
dftest.to_csv('2018_result.csv')
'''
#ML model: not used
'''
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def get_dataset(trainingset, validation_size):
	n = len(trainingset.columns)-1
	array = trainingset.values
	X = array[:,0:n]
	Y = array[:,n]
	seed = 7
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
	X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
	return X_train, X_validation, Y_train, Y_validation

le = preprocessing.LabelEncoder()
f = pandas.read_csv('feature.csv')
feature = f.columns.tolist()
le.fit(feature)
df['feature_code'] = le.transform(df.feature.values.tolist())
df = df[['feature_code','icd9']]

df['icd9'] = df.icd9.astype(str)
X_train, X_validation, Y_train, Y_validation = get_dataset(df, 0.2)
c = SVC()
c.fit(X_train, Y_train)
p = c.predict(X_validation)
cf = confusion_matrix(Y_validation, p)
print(cf)
cr = classification_report(Y_validation, p)
print(cr)
'''

















