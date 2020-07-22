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
