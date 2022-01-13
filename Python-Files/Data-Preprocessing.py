#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
from shutil import copyfile
import os

import sys
import time

import cv2
from skimage import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# ### Load Dataset
# 
# * `train-images-boxable-with-rotation.csv` file contains the image name and image url
# * `train-annotations-bbox.csv` file contains the bounding box info with the image name and the image label name
# * `class-descriptions-boxable.csv` file contains the image label name corresponding to its class name
# 
# Download link:
# 
# https://storage.googleapis.com/openimages/web/download.html

# In[2]:


get_ipython().system('wget https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv')


# In[3]:


get_ipython().system('wget https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv')


# In[4]:


get_ipython().system('wget https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv')


# In[5]:


images_fname = 'train-images-boxable-with-rotation.csv'
annotations_fname = 'train-annotations-bbox.csv'
class_descriptions_fname = 'class-descriptions-boxable.csv'


# In[6]:


images = pd.read_csv(images_fname)
images.sample()


# In[7]:


annotations_bbox = pd.read_csv(annotations_fname)
annotations_bbox.sample()


# 1. **XMin, XMax, YMin, YMax**: coordinates of the box, in normalized image coordinates.
# 2. **IsOccluded**: Indicates that the object is occluded by another object in the image.
# 3. **IsTruncated**: Indicates that the object extends beyond the boundary of the image.
# 4. **IsGroupOf**: Indicates that the box spans a group of objects (e.g., a bed of flowers or a crowd of people). We asked annotators to use this tag for cases with more than 5 instances which are heavily occluding each other and are physically touching.
# 5. **IsDepiction**: Indicates that the object is a depiction (e.g., a cartoon or drawing of the object, not a real physical instance).
# 6. **IsInside**: Indicates a picture taken from the inside of the object (e.g., a car interior or inside of a building).
# 

# In[8]:


class_descriptions = pd.read_csv(class_descriptions_fname, header=None)
class_descriptions.sample()


# ### Visualize Bounding Box

# In[9]:


def draw_bounding_box(img_id):
  img_url = images.loc[images["ImageID"] == img_id]['OriginalURL'].values[0]
  img = io.imread(img_url)

  height, width, channel = img.shape
  print(f"Image Shape: {img.shape}")

  bounding_boxes = annotations_bbox[annotations_bbox['ImageID'] == img_id]

  for index, row in bounding_boxes.iterrows():
      xmin, xmax, ymin, ymax = row['XMin'], row['XMax'], row['YMin'], row['YMax']

      #since the coordinates are normalized
      xmin, xmax, ymin, ymax = int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height)

      label_name = row['LabelName']
      class_series = class_descriptions[class_descriptions[0] == label_name]
      class_name = class_series[1].values[0]

      print(f"Coordinates: {xmin, ymin}, {xmax, ymax}")

      cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 5)
      font = cv2.FONT_HERSHEY_SIMPLEX
      cv2.putText(img, class_name, (xmin, ymin - 10), font, 3, (0, 255, 0), 5)
      
  plt.figure(figsize=(15,10))
  plt.title('Image with Bounding Box')
  plt.imshow(img)
  plt.axis("off")
  plt.show()


# In[10]:


for img_id in random.sample(list(annotations_bbox["ImageID"]), 1):
  draw_bounding_box(img_id)


# ### Get a Subset of the Dataset to use (since the original dataset contains ~1 million images)

# In[11]:


classes = list(np.random.choice(list(class_descriptions[1]), 10))
classes


# In[12]:


labels = []

for class_name in classes:
  row = class_descriptions[class_descriptions[1] == class_name]
  label_name = row[0].values[0]
  labels.append(label_name)


# In[13]:


bboxes = []

for label in labels:
  label_bbox = annotations_bbox[annotations_bbox['LabelName'] == label]
  bboxes.append(label_bbox)


# In[14]:


classes_removable_idx = set()

for class_name, bbox in zip(classes, bboxes):
  print(f'{len(bbox)} {class_name} in the dataset')
  if len(bbox) == 0:
    classes_removable.add(classes.index(class_name))


# In[15]:


img_ids = []

for bbox in bboxes:
  img_ids.append(bbox['ImageID'])


# In[16]:


unique_ids = []

for id in img_ids:
  unique_ids.append(np.unique(np.array(id)))


# In[17]:


min_count = float('inf')

for id, class_name in zip(unique_ids, classes):
  print(f'There are {len(id)} unique images with {class_name}')
  if len(id) > 0:
    min_count = min(min_count, len(id))

print(f'\n{min_count} images to be selected per class')


# In[18]:


n = min_count

selected_img_ids = []

for ids in img_ids:
  if len(ids) >= n:
    selected_img_ids.append(random.sample(list(ids), n))


# In[19]:


pds = []

for ids in selected_img_ids:
  curr_pd = images.loc[images['ImageID'].isin(ids)]
  pds.append(curr_pd)

pds[0].shape


# In[20]:


dicts = []

for df in pds:
  curr_dict = df[['ImageID', 'OriginalURL']].set_index('ImageID')["OriginalURL"].to_dict()
  dicts.append(curr_dict)


# In[21]:


mappings = [curr_dict for curr_dict in dicts]


# In[22]:


for idx in classes_removable_idx:
  labels.pop(idx)
  classes.pop(idx)


# ### Download Selected Images

# In[23]:


for idx, class_name in enumerate(classes):
  n_issues = 0

  if idx >= len(mappings):
    break

  if not os.path.exists(class_name):
    os.mkdir(class_name)

  for img_id, url in mappings[idx].items():

    try:
      img = io.imread(url)
      saved_path = os.path.join(class_name, img_id + ".jpg")
      io.imsave(saved_path, img)

    except Exception as e:
      n_issues += 1

  print(f"Images Issues for {class_name}: {n_issues}")


# ### Required Dataset Format
# 
# (img_path, xmin, xmax, ymin, ymax, class_name)
# 
# train: 0.8
# validation: 0.2

# In[24]:


train_path = 'train'
test_path = 'test'

get_ipython().system('mkdir train test')


# In[26]:


for i in range(len(classes)):
    
    all_imgs = os.listdir(classes[i])
    all_imgs = [f for f in all_imgs if not f.startswith(b'.')]
    random.shuffle(all_imgs)
    
    limit = int(n * 0.8)

    train_imgs = all_imgs[: limit]
    test_imgs = all_imgs[limit :]
    
    for j in range(len(train_imgs)):
        original_path = os.path.join(str.encode(classes[i]), train_imgs[j])
        new_path = os.path.join(str.encode(train_path), train_imgs[j])
        copyfile(original_path, new_path)
    
    for j in range(len(test_imgs)):
        original_path = os.path.join(str.encode(classes[i]), test_imgs[j])
        new_path = os.path.join(str.encode(test_path), test_imgs[j])
        copyfile(original_path, new_path)


# In[27]:


get_ipython().system('ls train | wc -l')


# In[28]:


get_ipython().system('ls test | wc -l')


# In[29]:


train_df = pd.DataFrame(columns=['FileName', 'XMin', 'XMax', 'YMin', 'YMax', 'ClassName'])

# Find boxes in each image and put them in a dataframe
train_imgs = os.listdir(train_path)
train_imgs = [name for name in train_imgs if not name.startswith('.')]

for i in range(len(train_imgs)):
    img_name = train_imgs[i]
    img_id = img_name[0:16]

    bbox_df = annotations_bbox[annotations_bbox['ImageID'] == img_id]
    for index, row in bbox_df.iterrows():
        labelName = row['LabelName']
        for i in range(len(labels)):
            if labelName == labels[i]:
                train_df = train_df.append({'FileName': img_name, 
                                            'XMin': row['XMin'], 
                                            'XMax': row['XMax'], 
                                            'YMin': row['YMin'], 
                                            'YMax': row['YMax'], 
                                            'ClassName': classes[i]}, 
                                           ignore_index=True)


# In[30]:


train_df.sample()


# In[32]:


train_img_ids = train_df["FileName"].sample().str.split(".").str[0].unique()


# In[33]:


for img_id in train_img_ids:
  draw_bounding_box(img_id)


# In[34]:


test_df = pd.DataFrame(columns=['FileName', 'XMin', 'XMax', 'YMin', 'YMax', 'ClassName'])

# Find boxes in each image and put them in a dataframe
test_imgs = os.listdir(test_path)
test_imgs = [name for name in test_imgs if not name.startswith('.')]

for i in range(len(test_imgs)):
    img_name = test_imgs[i]
    img_id = img_name[0:16]
    bbox_df = annotations_bbox[annotations_bbox['ImageID'] == img_id]
    for index, row in bbox_df.iterrows():
        labelName = row['LabelName']
        for i in range(len(labels)):
            if labelName == labels[i]:
                test_df = test_df.append({'FileName': img_name, 
                                            'XMin': row['XMin'], 
                                            'XMax': row['XMax'], 
                                            'YMin': row['YMin'], 
                                            'YMax': row['YMax'], 
                                            'ClassName': classes[i]}, 
                                           ignore_index=True)


# In[35]:


train_df.to_csv('train.csv')
test_df.to_csv('test.csv')


# ### Write train.csv to train_annotation.txt and test.csv to test_annotation.txt

# In[36]:


train_df = pd.read_csv('train.csv')

with open("train_annotation.txt", "w+") as f:
  for idx, row in train_df.iterrows():
      img = cv2.imread('train/' + row['FileName'])
      height, width = img.shape[: 2]
      x1, x2, y1, y2 = int(row['XMin'] * width), int(row['XMax'] * width), int(row['YMin'] * height), int(row['YMax'] * height)
      
      google_colab_file_path = 'drive/My Drive/AI/Dataset/Open Images Dataset v4 (Bounding Boxes)/train'
      fileName = os.path.join(google_colab_file_path, row['FileName'])
      className = row['ClassName']
      f.write(fileName + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + className + '\n')


# In[37]:


test_df = pd.read_csv('test.csv')

with open("test_annotation.txt", "w+") as f:
  for idx, row in test_df.iterrows():
      img = cv2.imread('test/' + row['FileName'])
      height, width = img.shape[:2]
      x1, x2, y1, y2 = int(row['XMin'] * width), int(row['XMax'] * width), int(row['YMin'] * height), int(row['YMax'] * height)
      
      google_colab_file_path = 'drive/My Drive/AI/Dataset/Open Images Dataset v4 (Bounding Boxes)/test'
      fileName = os.path.join(google_colab_file_path, row['FileName'])
      className = row['ClassName']
      f.write(fileName + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + className + '\n')


# ###Copy to the Google Drive folder

# In[38]:


from google.colab import drive
drive.mount('/content/drive')


# In[43]:


get_ipython().system('mv /content/train /content/drive/My\\ Drive/AI/Dataset/Open\\ Images\\ Dataset\\ v4\\ \\(Bounding\\ Boxes\\)')


# In[44]:


get_ipython().system('mv /content/test /content/drive/My\\ Drive/AI/Dataset/Open\\ Images\\ Dataset\\ v4\\ \\(Bounding\\ Boxes\\)')


# In[46]:


get_ipython().system('mv /content/train_annotation.txt /content/drive/My\\ Drive/AI/Dataset/Open\\ Images\\ Dataset\\ v4\\ \\(Bounding\\ Boxes\\)')


# In[45]:


get_ipython().system('mv /content/test_annotation.txt /content/drive/My\\ Drive/AI/Dataset/Open\\ Images\\ Dataset\\ v4\\ \\(Bounding\\ Boxes\\)')

