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
### Purpose of Project

Classification of consumer's complaints from .csv file. It is simple text classification study. A sample output is shown below.

![GitHub Logo](https://github.com/nursultanbolel/Classification-Of-Consumer-Finance-Complaints/blob/master/images/sample_output.PNG)

### Dataset
- Each week the CFPB sends thousands of consumers’ complaints about financial products and services to companies for response. Those complaints are published here after the company responds or after 15 days, whichever comes first. By adding their voice, consumers help improve the financial marketplace.

- I used .csv file  you can find there [Kaggle](https://www.kaggle.com/cfpb/us-consumer-finance-complaints)

- Information of .csv file is shown below in Spyder IDE.

![GitHub Logo](https://github.com/nursultanbolel/Classification-Of-Consumer-Finance-Complaints/blob/master/images/dataframe_inf.png)

- I used just two columns that are 'product' and 'consumer_complaint_narrative'. The aim is to predict the product according to consumer narrative.

- 'product' was including 13 different values. I joined same department and then now it has 10 different value. These values' information is shown below.

![GitHub Logo](https://github.com/nursultanbolel/Classification-Of-Consumer-Finance-Complaints/blob/master/images/product_inf.png)

