# Point-Annotation-based-Crowd-Counting-and-Detection (Nov 2021)
This project designs a model that performs Crowd Counting & Detection using Point Annotations on the Person heads. Existing systems are incapable of performing counting & detection simultaneously. Regression-based systems use density map for crowd counting but they do not have the potential for person detection. On the other hand, Detection-based systems use bounding box annotations on person heads but they are computationally too expensive to use. With modernization, we are being watched almost everywhere and everytime by CCTV cameras etc. This emphasizes the need for crowd counting & detection simultaneously in a more reliable & less expensive manner. The hybrid model designed here uses bounding boxes if they are available and generate pseudo ground truth bounding boxes from point annotations if they aren't available. 

## Scope
1. Safety monitoring
2. Behaviour modeling
3. Survey purposes
4. Person tracking & recognition

## Architecture
<img width="857" alt="Architecture" src="https://user-images.githubusercontent.com/28973352/149382342-25492329-aa2d-4962-a72b-6f0ee9246fe0.png">

## Base Model
<img width="589" alt="Base-Model" src="https://user-images.githubusercontent.com/28973352/149383466-32a1f87e-fb22-4a50-8368-18a47a27aabf.png">

## Loss functions
1. Classification Loss - Object/Person classification
2. Regression Loss - When the dataset includes bounding boxes
3. Locally Constrained Regression Loss - When pseudo ground truth bounding boxes have been generated 

## Datasets
1. Google’s Open Images Dataset V6 - Object detection (with bounding box annotations)
https://storage.googleapis.com/openimages/web/download.html
2. NWPU-Crowd - Crowd counting and detection (using only point annotations)
https://gjy3035.github.io/NWPU-Crowd-Sample-Code/
3. RGBT Crowd Counting - Crowd counting and detection (using only point annotations)
http://lingboliu.com/RGBT_Crowd_Counting.html

## Empirical Results
<img width="827" alt="Results" src="https://user-images.githubusercontent.com/28973352/149384487-1eafb6d6-e067-44c1-ab61-556605f49abf.png">

## Execution Instructions
1. Data-Preprocessing.py helps you to understand how the images are preprocessed & how several parts of image are identified.
2. Data-Preprocessing-People-Only.py helps you to preprocess images focusing on identifying only the persons.
3. Training-and-Testing.py helps you to understand how the model is trained & tested for different classes.
4. Training-and-Testing-People-Only.py helps you to perform actual training & testing of the model aiming to detect as well as count people.

## Languages, Tools & Technolgies
1. Python
2. Google Colab
3. Tenserflow
4. Keras
