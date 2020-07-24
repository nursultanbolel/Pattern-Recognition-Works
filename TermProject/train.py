from joblib import dump
import numpy as np
import nltk
from nltk.corpus import stopwords
import pandas as pd
import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import svm
from string import digits

df = pd.read_csv('train.csv')
df = df[pd.notnull(df['consumer_complaint_narrative'])]

STOPWORDS=stopwords.words('english')
stopwords_extra=['bank', 'america', 'x/xx/xxxx', '00']
STOPWORDS.extend(stopwords_extra)
replace_espaco = re.compile('[/(){}\[\]\|@,;]')
df = df.reset_index(drop=True)

def pre_processamento(text):
    text = text.lower()
    text = replace_espaco.sub(' ', text)
    remove_digits = str.maketrans('', '', digits)
    text = text.translate(remove_digits)
    text = text.replace('x', '')
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

df['consumer_complaint_narrative'] = df['consumer_complaint_narrative'].apply(pre_processamento)

vocab = set()
doc_vocab = []
number_of_terms = 0
number_of_docs = 0
class_dictionary = {}
cls_index = 0
doc_clss_index = []
count_of_that_class = [] 
class_name=[]

for i in range(df.shape[0]):
    number_of_docs +=1
    text = df.iloc[i]['consumer_complaint_narrative']
    cls = df.iloc[i]['product']
    if (class_dictionary.get(cls))==None:
        class_dictionary[cls]=cls_index 
        tmp = cls_index
        cls_index+=1
        count_of_that_class.append(1)
        class_name.append(cls)
    else:
        tmp = class_dictionary[cls]
        count_of_that_class[tmp] +=1      
    doc_clss_index.append(tmp)
    tokens= nltk.word_tokenize(text)
    tmp_set = set()
    number_of_terms += len(tokens)
    for word in tokens:
        vocab.add(word)
        tmp_set.add(word)
    doc_vocab.append(tmp_set)
    
np.save('class_dictionary',class_dictionary)  
np.save('class_name',class_name)    

print("\nDOCUMENT INFORMATIONS:")
print("vocab size:", len(vocab))
print ("number of terms (all tokens):", number_of_terms)
print ("number of docs:", number_of_docs)
print ("number of classes:", cls_index)

vocab_size = len(vocab)
number_of_classes = cls_index
count_of_that_class = np.asarray(count_of_that_class)
probability_of_classess = count_of_that_class / number_of_docs

print("\nCLASS PROBABILITIES:")
tmp_view = pd.DataFrame(probability_of_classess)
print(tmp_view)

word_occurance_frequency = np.zeros(vocab_size, dtype=int)
word_occurance_frequency_vs_class = np.zeros((vocab_size, number_of_classes), dtype=int)
word_index = {}
counter = -1
vocab_list = []

for word in vocab:
    counter+=1
    word_index[word]=counter
    vocab_list.append(word)
vocab_list = np.asarray(vocab_list)

for i in range(0, number_of_docs):
    for word in doc_vocab[i]:
        index = word_index[word]
        word_occurance_frequency[index]+=1
        word_occurance_frequency_vs_class[index][doc_clss_index[i]] +=1

p_w = word_occurance_frequency/number_of_docs
p_w_not = 1 - p_w
p_c = probability_of_classess

p_class_condition_on_w = np.zeros((number_of_classes, vocab_size), dtype=float)
tmp = word_occurance_frequency_vs_class.T
for i in range(0, number_of_classes):
    p_class_condition_on_w[i] = tmp[i]/word_occurance_frequency

p_class_condition_on_not_w = np.zeros((number_of_classes, vocab_size), dtype=float)
for i in range(0, number_of_classes):
    p_class_condition_on_not_w[i] = (count_of_that_class[i]-tmp[i])/(number_of_docs-word_occurance_frequency)
    
word_ig_information = []
e_0 = 0.0
for c_index in range(0, number_of_classes):
    e_0+=p_c[c_index]*np.log2(p_c[c_index])
e_0 = -e_0
for w_index in range(0,vocab_size):
    e_1 = 0.0
    for c_index in range(0, number_of_classes):
        tmp1 = p_class_condition_on_w[c_index][w_index]
        if tmp1 !=0:
            e_1 += p_w[w_index]*tmp1*np.log2(tmp1)
        tmp2 = p_class_condition_on_not_w[c_index][w_index]
        if tmp2 !=0:
            e_1 += (1-p_w[w_index])*(tmp2*np.log2(tmp2))
    e_1 = -e_1
    
    information_gain = e_0 - e_1
    word_ig_information.append([information_gain, vocab_list[w_index]])
    
word_ig_information = sorted(word_ig_information, key=lambda x: x[0], reverse=True)
features = pd.DataFrame(word_ig_information)
features.columns=['information_gain', 'word']
features = features.head(200)
features.to_csv("information_gain.csv", index=False)

document_dicts= []
for i in range(df.shape[0]):
    text = df.iloc[i]['consumer_complaint_narrative']
    # cls = df.iloc[i]['product']       
  
    tokens= nltk.word_tokenize(text)
    tmp_dict = {}
    for word in tokens:
        if (not (word in tmp_dict)):
            tmp_dict[word]=1
        else:
            tmp_dict[word]+=1
    document_dicts.append(tmp_dict)

# features = pd.read_csv('information_gain.csv')

features = np.asarray(features)
feature_size = len(features)
X_count = np.zeros((number_of_docs, feature_size), dtype=float)
for i in range(0, number_of_docs):
    for j in range(0, feature_size):
        tmp_dict= document_dicts[i]
        tmp_word = features[j][1]
        if tmp_word in tmp_dict:
            X_count[i][j]=tmp_dict[tmp_word]
        else:
            X_count[i][j]=0
tmp_sum = np.array([np.sum(X_count, axis=1)])

X = np.zeros(shape = (number_of_docs, feature_size))
tmp_sum = tmp_sum.T

for i in range(X_count.shape[0]):
    if (tmp_sum[i] == 0):
        X[i,:] = 0 
    else:
        X[i,:] = X_count[i,:]/tmp_sum[i]

y = np.asarray(doc_clss_index)

k=5
kf = KFold(n_splits=k, shuffle=True)
clf = svm.LinearSVC()

shuffled_index = np.arange(0, number_of_docs)
np.random.shuffle(shuffled_index)

for train_index, test_index in kf.split(X):
    train_index_shuffled = np.take(shuffled_index, train_index)
    test_index_shuffled = np.take(shuffled_index, test_index)
    X_train, X_test = X[train_index_shuffled], X[test_index_shuffled]
    y_train, y_test = y[train_index_shuffled], y[test_index_shuffled]
    
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    
    print("\nSVM Model Accuracy: ", accuracy_score(y_test, prediction))
    dump(clf, 'svm_model.joblib') 