# Handwriting Identification

* This repository holds an attempt to apply transfer learning techniques and convolutional neural nets (CNNs) to model and predict the identity of the writer of a given handwritten image from the [CSAFE Handwriting Database (version 1)](https://data.csafe.iastate.edu/HandwritingDatabase/?saveQueryContent=handwritingdbstudy-%3E++%28Writer_ID+%3C%3D+%270090%27%29+&files%5B%5D=&study=handwritingdbstudy&left-operands-parameters-name=Writer_ID&filter-operators-name=%3D&right-operands-parameters-value=Writer_ID&paramValues=0009#). 

## Overview

This project explores the application of CNNs to the problem of handwriting-based writer identification using the CSAFE Handwriting Database (version 1). The goal was to build a multi-class classification model that could accurately predict which of 90 writers produced a given handwritten sample image. The problem was framed as a supervised image classification task with 90 distinct classes, one for each writer. The initial approach involved training a ResNet50V2 model as a baseline. However, issues with training time, loss instability, and poor performance eventually led to experimentation with smaller, more efficient architectures; namely, MobileNetV2 and EfficientNetB1. Data augmentations (random cropping, contrast, and rotation) were tested but ultimately found to degrade performance in preliminary models. A stratified 80/20 train-validation split was used to ensure class balance, and model training was primarily done in Google Colab for efficiency. Performance was evaluated using classification accuracy and macro-averaged ROC AUC. While the ResNet-based models struggled to exceed 5% accuracy, they still showed decent AUC scores. Both MobileNet and EfficientNet achieved markedly improved results (especially during training). MobileNetV2, in particular, attained a training accuracy of approximately 40% and a macro-average AUC of 0.96, making it the strongest performing model. 


## Summary of Work Done

### Data

* Type: Image data, scans of handwriting samples; .csv file containing metadata on each writer is also included when downloading the data.
* Size: 2430 images, 1.18 GB, 90 writers (i.e., classes) & 27 images each
* Instances: an 80/20 train/validation split was used, with stratified sampling to ensure all classes were equally represented in each set; validation data was used for testing as well
*Note: There are multiple "versions"/updates to this dataset; version 1 was used for this attempt.*

#### Preprocessing / Clean Up
The data required minimal processing. Python scripts were used to organize the image files in directories based on writer IDs, and then further organized into larger train/validation directories (outlined in `reload_data.py`). Image augmentations utilized include random crop (in order to have more focus on writing patterns and get "more" data per person), random contrast (to add variance in "pen strokes"), and random rotations (at small increments). 


#### Data Visualization

Examples of given image data.

![](/Images/example_images.png)


Examples of applied augmentations.

![](/Images/example_augmentation2.png)

![](/Images/example_augmentation.png)


### Problem Formation

The input would be any given image, the output would be the predicted writer/class it belonged to. The initial model trained was a ResNet50V2 model, supplemented to support a multi-class problem. This was the first pick because it seemed to balance size, performance, and time investments. Other models trained were EfficientNetB1 and MobileNetV2. These were primarily chosen because they were smaller and faster than ResNet while promising similar results. The adam optimizer function was used for all 3 models. For most models, the image size used was (384, 384) and the batch size was set at 18; this is exceptioned by the very first model, run in a local environment, with image sizes at (180, 180) and batch sizes of 8.


### Training

The original attempt (in local Jupyter environment) trained a ResNet50V2 model on 10 epochs with an image size of (224,224) and batch sizes of 8. This model trained relatively quickly, in about 12 minutes.
With Colab's environment and resourcing, the image_size was increased to (384, 384) and batch size to 18. These hyperparameters were used for the remainder of the models.
The Base ResNet model was trained with 8 epochs. This training session took especially long, taking over 20 mins per epoch. This model was supplemented with augmented image data and trained for an additional 12 epochs (which took approximately 6 mins per epoch).
The other models utilized (i.e., MobileNetV2 and EfficientNetB1) trained with a similar schedule to the augmented model; 10 epochs each. Training was typically stopped earlier than was ideal (i.e., before loss could plateau), at an arbitrarily set limit, due to time and processing constraints. Initial difficulties, including lost runtime and significantly long training durations, were mitigated mostly through limiting the number of epochs, prefetching the data, switching to Google Colab from local/Jupyter notebook for the brunt of training, as well as eventually exploring smaller models.

Base ResNet model loss curve

![](/Images/BaseModel_loss.png)


Augmented model loss curve (noticeably inefficient)

![](/Images/Augmented_loss.png)


EfficientNet loss curve

![](/Images/EfficientNet_loss.png)


MobileNet model loss curve

![](/Images/MobileNet_loss.png)




### Performance Comparison

The primary performance metric used during training was accuracy. Since this was a 90-class identification problem, baseline/"random chance" accuracy was 0.011. ResNet-based models did relatively poorly, rarely passing 0.05 accuracy during training. In fact, the Augmented model's accuracy *decreased*, which discouraged use of data augmentation of future models. However, due to MobileNet and EfficientNet's seeming proficiency, it would be interesting to test the effect of augmentations on these non-ResNet models. These latter two models each achieved training accuracies of approximately 0.4. 

ROC curve analysis comparison was also taken, the primary metric being the macro-average between all classes (since all classes are evenly represented). When comparing Base ResNet vs. Augmented model, the Base model's AUC score was greater. This also contributed to dismissing the value of the current augmentations. 

Base Model vs. Augmented Model ROC Curves

![](/Images/BaseVsAugment_ROC.png)


A ROC curve comparison was also done between the Base ResNet model, EfficientNet, and MobileNet. The highest performing model being MobileNet (macro-average AUC = 0.96), EfficientNet performed the worst of the three (macro-average AUC = 0.83).


Model ROC Curve Comparison

![](/Images/Model_Comparison_ROC.png)


It should be noted that due to the limited nature of the dataset, the same data were used for both validation and testing sets, introducing the possibility of leakage, which could inflate metrics/results.


### Conclusions

This project was an exploration of writer identification as an image classification problem using CNNs. While the task initially seemed approachable, especially with promising pre-trained architectures like ResNet50V2, practical challenges quickly emerged, including long training times, unstable loss behavior, and difficulty tuning for 90 unique classes. These obstacles ultimately limited the utility and training potential of the models and made it clear that deeper, heavier models are not always the best fit. Lighter models like MobileNetV2 and EfficientNetB1 trained faster, introducing more possibility for improvement, outperformed ResNet50V2 in accuracy and performed equally and/or better with concern to AUC metrics. Ultimately, it was MobileNetV2 that yielded the best results. 


### Future Work

In the future, I'd like to determine the reason behind ResNet's apparent issues during training and potentially correct and re-train it. To further develop my models, I'd like to upsize by using the updated/larger versions of the CSAFE Handwriting Database, as well as try a larger variety of models, more epochs (i.e., greater time and processing investment), and apply more augmentations. One concern I'd specifically like to address is resolution and loss of data through resizing and/or cropping; to achieve this, I'd like to attempt pre-cropping images, before images are resized, (also serves to inflate class sizes) and using boundary boxes to detect text to ensure avoidance of excessive whitespace. Ideally, this would also yield a volume of data better suited toward a clean training/validation/testing split.

Another avenue of interest is investigating the minimal amount of layers/neurons/image sizes required to achieve significant, non-trivial results. Additionally, exploring dimensionality and/or feature reduction and identifying possible "writer-specific" features is another aspect of note.


## How to Reproduce Results

To reproduce the analysis and modeling results:
* Clone this repository.
* Open `Colab_Vision_Baseline&Augmentations.ipynb` using Google Colab
* Download CSAFE dataset and upload it to the virtual Colab environment
* Download and upload `reload_data.py` to the virtual Colab environment
* Run all cells
* Repeat with `Colab_CompareAugmentation.ipynb`, `Colab_ExtraModels_Train.ipynb`, and `Colab_Compare_Models.ipynb`


**Note: All steps/modeling can be done locally (e.g., with Jupyter), but this would require modification of existing modules/notebooks/code and proper hardware.* 


### Contents of Repository
* Vision_Feasibility_ZS.ipynb: Jupyter notebook containing initial exploration of data, including preliminary dataloading setup (refined in future notebooks and `reload_data.py` module)
* Vision_TrainBaseModel.ipynb: Jupyter notebook containing initial baseline model (ResNet50V2), includes unsuccessful attempts to stratify data; model was trained on unstratified data
* Colab_Vision_BaselineAttempt.ipynb: shows results from the first Colab model (attempting to establish a baseline model with stratified train/validation sets); runtime disconnected and was lost
* Colab_Baseline&Augmentations.ipynb: reattempting what was started in `Colab_Vision_BaselineAttempt.ipynb` along with training on augmented data
* Colab_CompareAugmentation.ipynb: comparing the performance of the Base model (ResNet50V2) with the one supplemented with augmented image data
* Colab_ExtraModels_Train.ipynb: training additional models, EfficientNetB1 and MobileNetV2
* Colab_Compare_Models.ipynb: comparing the performance of the Base ResNet model with the EfficientNet and MobileNet models trained in `Colab_ExtraModels_Train.ipynb`
* reload_data.py: module containing majority of "preprocessing" steps for reloading (includes functions for reorganizing into train/validation directories, visualizing augmentations, etc.) data within Colab environment


### Software Setup
Google Colab was used for majority of model training for its computational processing resources. Visualizations were completed with matplotlib. Modelling and analysis was done through tensorflow, keras, numpy, and scikit-learn. Data organization was automated with the os, shutil, and zipfile modules. 


### Data

The data can be downloaded on its [CSAFE database webpage](https://data.csafe.iastate.edu/HandwritingDatabase/?saveQueryContent=handwritingdbstudy-%3E++%28Writer_ID+%3C%3D+%270090%27%29+&files%5B%5D=&study=handwritingdbstudy&left-operands-parameters-name=Writer_ID&filter-operators-name=%3D&right-operands-parameters-value=Writer_ID&paramValues=0009#).


## Citations

Crawford, Amy; Ray, Anyesha; Carriquiry, Alicia; Kruse, James; Peterson, Marc (2019): CSAFE Handwriting Database. Iowa State University. Dataset. https://doi.org/10.25380/iastate.10062203.v1
