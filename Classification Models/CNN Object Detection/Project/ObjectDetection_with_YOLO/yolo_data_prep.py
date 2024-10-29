import pandas as pd
import numpy as np
import seaborn as sns
import os
import shutil
import xml.etree.ElementTree as ET
import glob

import json

# XML을 YOLO에 맞게 포맷을 변환하기
# 해당 기능은 https://towardsdatascience.com/convert-pascal-voc-xml-to-yolo-for-object-detection-f969811ccba5

def xml_to_yolo_bbox(bbox, w, h):

  x_center = ((bbox[2] + bbox[0]) / 2) / w
  y_center = ((bbox[3] + bbox[1]) / 2) / h
  width = (bbox[2] - bbox[0]) / w
  height = (bbox[3] - bbox[1] ) / h
  return [x_center, y_center, width, height]

def create_folder(path):
  if not os.path.exists(path):
    os.makedirs(path)

create_folder('yolo7\\train\\images')
create_folder('yolo7\\train\\labels')
create_folder('yolo7\\val\\images')
create_folder('yolo7\\val\\labels')
create_folder('yolo7\\test\\images')
create_folder('yolo7\\test\\labels')

img_folder = 'images'

_, _, files = next(os.walk(img_folder))
pos = 0
for f in files:
  source_img = os.path.jsoin(img_folder, f)
  if pos < 700:
    dest_folder = 'yolov7\\train'
  elif (pos >=700 and pos < 800):
    dest_folder = 'yolov7\\val'
  else:
    dest_folder = 'yolov7\\test'
  destination_img = os.path.join(dest_folder, 'images')
  shutil.copy(source_img, destination_img)

  label_file_basename = os.path.splitext(f)[0]
  label_source_file = f"{label_file_basename}.xml"
  label_dest_file = f"{label_file_basename}.txt"

  label_source_path = os.path.join('annotations', label_source_file)
  label_dest_path = os.path.join(dest_folder, 'labels')
  if os.path.exists(label_source_path):
    # xml 파일 내용 불러오기
    tree = ET.parse(label_source_path)
    root = tree.getroot()
    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)
    classes = ['with_mask', 'without_mask', 'mask_wear_incorrect']
    result = []
    for obj in root.findall('object'):
      label = obj.find("name").text
      index = classes.index(label)
      pil_bbox = [int(x.text) for x in obj.find("bn")]
      yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
      bbox_string = " ".join([str(x) for x in yolo_bbox])
      result.append(f"{index} {bbox_string}")
      if result:
        with open(label_dest_path, 'w', encoding=utf-8) as f:
          f.write("\n".join(result))
    pos += 1
  



