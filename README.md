# Pattern-Recognition-Works
includes basic pattern recognition works

All homeworks directory include description which is .pdf file.

## HW-1 
Fundamental operations about image processing like dominanting green,hue,value channel  and resizing is coded. I didn't use present functions, I wrote my functions. Output image is has four quadrants:  <br>
 - The top left quadrant is the original image,<br>
 - The top right quadrant is the Green image, <br>
 - The bottom left quadrant is the Value image,<br>
 - The bottom right quadrant is the Hue image<br>
 
 You can see an example of output image: <br>
 ![GitHub Logo](https://github.com/nursultanbolel/Pattern-Recognition-Homeworks/blob/master/HW-1/savedCatImage.jpg)
 
## HW-2
I used KNN algorithm and cosine similarity distance to classify images from Cifar-10 dataset. There are four basic steps:<br>
Step 1: Download the Cifar-10 dataset python version.<br>
Step 2: Convert images to vector format, all images(32x32x3) are converted to 1x3072 vectors.<br>
Step 3: Split the dataset(train: 50.000 lines, test: 10.000 lines)<br>
Step 4: Use cosine similarity distance to compute similarity

You can see the most similar class name on console screen.

## HW-3
I used Maholanabis distance in place of Bayes Classifier to classify images from Cifar-10 dataset.<br>
The dataset includes 10 classes and each class has 5000 lines.<br>
I assumed that class probabilities are equal. p(w0)=p(w1)=,...,=p(w9) <br> 
According to the Bayes Rule, I calculated the distances of given test sample in order to check whether belongs or not to a specific class. <br>
Minimum distance indicates predicted class label of the processed test sample. <br>

You can see the distances to all classes and predicted class name on console screen.

## HW-4
I used SVM classifier to classify multi classes. Dataset is Caltech-101 that has 15 classes and each class has different number of samples. <br>
You can find dataset in the repo as .rar file format. <br>
The aim is to use SVM classifier, one-against-all methodology, in order to find different hyperplanes that is capable to separate classes. There are three basic steps: <br>
Step 1: All images' size in the dataset is 128x128x3. I converted to a vector (1x49152). <br>
Step 2: I extracted 512 HOG features from the images by using sklearn. After the feature extraction the training data (1457x49152) are represented as (1457x512) matrix form. <br>
Step 3: I used OneVsTestClassifier() function which is in sklearn to have 15 hyperlanes for train step.<br>

You can see performance measures and classification of test sample in the pictures below from console screen:  
![GitHub Logo](https://github.com/nursultanbolel/Pattern-Recognition-Works/blob/master/HW-4/performance_measures.JPG) <br>
![GitHub Logo](https://github.com/nursultanbolel/Pattern-Recognition-Works/blob/master/HW-4/test_sample.JPG) <br>

## HW-5
In this homework I added feature selection step to the HW-4. <br>
After feature selection, I chose more meaningful and rich features. I used scikit-learn library provides the SelectKBest class to select a specific number of features.I selected features according to the k highest scores. 
* score_func: Chi-Squared
* k: number of top features to select.
* selected_features: holds indices of selected features <br>
<code> test = SelectKBest(score_func=chi2, k=256) </code>

You can see performance measures and classification of test sample in the pictures below from console screen:  
![GitHub Logo](https://github.com/nursultanbolel/Pattern-Recognition-Works/blob/master/HW-5/performance_measures.JPG) <br>
![GitHub Logo](https://github.com/nursultanbolel/Pattern-Recognition-Works/blob/master/HW-5/test_sample.JPG) <br>

## TERM PROJECT(Classification Of Consumer Finance Complaints Using Information Gain)
I tried to classify consumer finance complaints in one of my repositories using deep learning. You can reach [there](https://github.com/nursultanbolel/Classification-Of-Consumer-Finance-Complaints). <br>

In this repository I tried to classify consumer finance complaints by calculating Information Gain. <br>

### PURPOSE
Main aim is to classify consumers’ complaints about financial according to bank departments like mortgage, credit card …These complaints can be classified without any person and forwered to related bank employee. Classification of problems can help the employees to reduce complaint solving time. Classifying complaints is a problem of text classification In this project I used pattern recognition method to classify. <br>

### DATASET
Each week the CFPB sends thousands of consumers’ complaints about financial products and services to companies for response. Those complaints are published after the company responds or after 15 days, whichever comes first. By adding their voice, consumers help improve the financial marketplace. Some these complaints informations are published in kaggle as text data in .csv file format. I used this dataset that is a .csv file. You can find there [Kaggle](https://www.kaggle.com/cfpb/us-consumer-finance-complaints) <br>

The dataset has 18 columns but I used just ‘product’ and ‘consumer_complaints_narrative’. Column ‘product’ holds department information and column ‘consumer_complaints_narrative’ holds complaints informations. ı used 157865 lines with consumer complaints. These columns are text format. My aim was to classify complaints according to department. You can see all columns and number of non-null lines at image below. <br>

- Total number of lines: 777959
- Number of lines with consumer complaints: 157865

![GitHub Logo](https://github.com/nursultanbolel/Pattern-Recognition-Works/blob/master/TermProject/images/dataFrame_information.png) <br>

Product column had more than10 classes and value count of classes are not equel. Before training step I joined some related classes like ‘Credit card’ -‘Prepaid card’ and I joined ‘Virtual currency’ – ‘Other financial service’ has few lines. You can see class names and number of data lines at image below. <br>

![GitHub Logo](https://github.com/nursultanbolel/Pattern-Recognition-Works/blob/master/TermProject/images/product_value_count.png) <br>

I wrote prepareDataset.py to join related columns, select needen two columns, shuffle and split dataset. You can see dataframe head information at image below. The dataframe has two columns and 157865 non empty lines. I suffled and then splited the dataset for train 90% for test 10%.

![GitHub Logo](https://github.com/nursultanbolel/Pattern-Recognition-Works/blob/master/TermProject/images/dataFrame_head.png) <br>

### METHOD
I used Information Gain metric to find important words for text classification. ı used 157865 lines text data for train and test model. Each complaint can be represented by the set of its words. But some words are more important and has more effect and more meaning. These words can be used for determining the context of a complaints.
In this part, I tried to find a set of 200 words that are more informative for document classification. <br>

###### Step 1: Preprocessing
Most complaints have stopwords and these words do not effect classification. I removed this words. I used nltk python library for stopword list. I also added some words stopwords list like ‘bank’ and ‘america’. I removed all numbers, punctions and  special characters. I convert all text to lowercase to reduce number of words. 

###### Step 2: Information Gain Metric
The dataset has 88146 different words, 13771601 tokens at 142078 lines. You can see informations about vocab size, number of tokens, documents and classes at image below. I calculated information gain metric for 88146 words. 

![GitHub Logo](https://github.com/nursultanbolel/Pattern-Recognition-Works/blob/master/TermProject/images/document_informations.png) <br>

You can see Information Gain formula at image below. <br>

![GitHub Logo](https://github.com/nursultanbolel/Pattern-Recognition-Works/blob/master/TermProject/images/IG_formula.png) <br>

I selected 200 words with the highest information gain value as a fature set. You can see top 10 features has highest information gain value at image below. Before training step I converted all complaints text to numbers using informain gains values.

![GitHub Logo](https://github.com/nursultanbolel/Pattern-Recognition-Works/blob/master/TermProject/images/featureSet_top_10.png) <br>

###### Step 3: Train(SVM Model)
I trained my model with the dataset by SVM from sklearn with  K-fold = 5. You can see results about accuracy values for each fold at image below. Almost every fold has same accuracy value.

![GitHub Logo](https://github.com/nursultanbolel/Pattern-Recognition-Works/blob/master/TermProject/images/svm_model_acc.png) <br>

###### Step 4: Testing Model(SVM Model)
I gave some examples to the SVM model. You can see two given complaints and about expected and pretrained classes informations at image below. In both complaints SVM model trully predicted.

![GitHub Logo](https://github.com/nursultanbolel/Pattern-Recognition-Works/blob/master/TermProject/images/sample_test.png) <br>

### PERFORMANCE MEASURES
I splitted 10% of the dataset for testing. You can see precision, recall and f-1-score values and support value means number of lines belong to each class at image below. This figure shows the SVM model’s accuracy equals to 81% for 10 classes. 

![GitHub Logo](https://github.com/nursultanbolel/Pattern-Recognition-Works/blob/master/TermProject/images/performance_measures.png) <br>

I calculated TPR, FPR , accuracy  and ROC curve for each class, you can see values as a table at image below. The SVM model accuracy is 81%. This value is good generally. But I reliased that each class has very different accuracy value. Class6(Mortgage)’s accuracy is 95% and Class9(Other financial service)’s acuracy is 50%. The reason for this is probably all classes have different number of lines. ‘Other financial service’ has just 288 lines in train dataset  and ‘Mortagage’ has 32000 lines in train dataset . At the same time Class7 and Class 8 accuracy values are lower than 70% because of the same reason.
In my opinion when I add some lines for Class7,Class8 and Class9 to train dataset I will get better accuracy for these three classes. 

![GitHub Logo](https://github.com/nursultanbolel/Pattern-Recognition-Works/blob/master/TermProject/images/classes_acc.PNG) <br>
![GitHub Logo](https://github.com/nursultanbolel/Pattern-Recognition-Works/blob/master/TermProject/images/class_info.png) <br>

### SETUP
I studied on laptop with following features at all steps 
- CPU: model i7-4510U, speed 2.00 GHz, cache 4 MB  
- Memory(RAM): memory 8 GB, frequency 1600 MHz, type DDR3L 
- GPU: brand NVIDIA, model GeForce 840M, memory 4 GB

I used Anaconda Spyder IDE
#packages in working  environment
#Name__Version
joblib__0.14.1
numpy__1.18.1
nltk __ 3.4.5
pandas__1.0.3
scikit-learn__0.22.1
spyder__4.1.3
