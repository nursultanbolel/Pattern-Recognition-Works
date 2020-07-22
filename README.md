Convert images to vector format# Pattern-Recognition-Works
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
I used KNN algorithm to classify images from Cifar-10 dataset. There are four basic steps:<br>
Step 1: Download the Cifar-10 dataset python version.<br>
Step 2: Convert images to vector format, all images(32*32*3) are converted to 1*3072 vectors.<br>
Step 3: Split the dataset <br>
