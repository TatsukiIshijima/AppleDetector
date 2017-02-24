# AppleDetector
Apple detect used by Texture feature

## Demo  
**Preparation**    
Move apple images and non apple images to apples directory.  

**Step1  Create patch xml file**  
Specify apple region and non apple region used imglab.  
Create applePatch.xml and nonapplePath.xml. (This xml file set in apples directory)

**Step2 Extract feature**  
Run this command(Create apple feature(Ture) and non apple feature(False))  
*python extract.py image_dir xxx.xml xxx.csv label*  
image_dir is an apple images directory.    
xxx.xml is created file in step1.  
Feature is saved at xxx.csv. File name is arbitrarily.  
Lable is 0 or 1  
After Created True feature and False feature, These file connected used Excel etc (add row).  

**Step3 Train feature**  
Run this command  
*python train.py xxx.csv*  
xxx.csv is created file in step2.  
train model is saved in trainModel directory.  

**Step4 Detect apple**  
Run this command  
*python detector.py xxx.model xxx.jpg size*  
xxx.model is created file in step3.  
xxx.jpg is test image.  
size is patch size.  

## Requirement  
1. Python3(Anaconda)  
2. dlib + imglab  
3. OpenCV2 or OpenCV3  
4. Numpy(Anaconda)  
5. Scikit-Learn(Anaconda)  
6. Scikit-Image(Anaconda)  
7. matplot(Anaconda)  
8. BeautifulSoup(Anaconda)  
