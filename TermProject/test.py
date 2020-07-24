from joblib import load
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.corpus import stopwords
import pandas as pd
import re
from sklearn.metrics import classification_report, roc_curve, auc
from string import digits

#read model, IG values
clf = load('svm_model.joblib') 
features = pd.read_csv('information_gain.csv')
features = np.asarray(features)

#read test dataset
df = pd.read_csv('test.csv')
df_before_preprocess = df
df = df[pd.notnull(df['consumer_complaint_narrative'])]
number_of_docs = df.shape[0]

#Preprocessing
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

#Class information(class number - class name)
class_name = np.load('class_name.npy')
class_dictionary = {}
number_of_class = class_name.shape[0]
print ('\n#####################\n# CLASS INFORMATION #\n#####################')
print('class number --> class name')
for i in range(number_of_class):
    print('\n',i,'-->',class_name[i])
    class_dictionary[class_name[i]]=i 

#Counting all words in each complaints(documents)    
document_dicts= []
doc_class_index = []
for i in range(number_of_docs):
    text = df.iloc[i]['consumer_complaint_narrative']   
    cls = df.iloc[i]['product']       
    doc_class_index.append(class_dictionary[cls])
    tokens= nltk.word_tokenize(text)
    tmp_dict = {}
    for word in tokens:
        if (not (word in tmp_dict)):
            tmp_dict[word]=1
        else:
            tmp_dict[word]+=1
    document_dicts.append(tmp_dict)
    
#Convering each complaint(documents) to numbers by using IG(information gain).
#Selecting features from all documnets and assigning numbers according to IG    
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

Y = np.asarray(doc_class_index)

#Model prediction
prediction = clf.predict(X)

#Some examples to see spyder console(complaint without preprocessing-expected class- redicted class)
print ('\n#############################\n# GIVING TEST DATA TO MODEL #\n#############################')
for i in range(4,10):
    print('\nCONSUMER COMPLAINT WITHOUT PREPROCESSING ===> "',df_before_preprocess.iloc[i]['consumer_complaint_narrative'],'"')
    print('EXPECTED CLASS ==>',class_name[int(Y[i])])
    print('PREDICTED CLASS ==>',class_name[prediction[i]])
 
#Performance Measures
print ('\n########################\n# PERFORMANCE MEASURES #\n########################\n')
print(classification_report(Y, prediction, target_names=class_name))

y_test = np.zeros(shape = (number_of_docs, number_of_class))
y_score = np.zeros(shape = (number_of_docs, number_of_class))

for i in range(number_of_docs):
    y_test[i,Y[i]] = 1;
    y_score[i,prediction[i]]=1;

fpr = dict()
tpr = dict()
roc_auc = dict()

#TPR,FPR,AUC values are calculated for each class
for i in range(0,number_of_class):
    print('\n---> CLASS ',class_name[i],'\n')
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    print('TPR :', tpr[i])
    print('FPR :',fpr[i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print('AUC : %0.2f' %np.float(roc_auc[i]))
    print('OneVsAll Approach ROC curve')
#Drawing ROC curve
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' %np.float(roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()