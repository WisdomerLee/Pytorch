# 사물인식 인식표 포맷
사물인식 알고리즘을 훈련시키기 위해 해당 그림에 사물이 어느 위치에, 어느 사물인지를 알려주는 파일의 포맷 형태

||Pascal VOC|COCO|YOLO|
|---|---|---|---|
|Full Name|Patterns Analysis, Statistical Modelling, and Computational Learning Visual Object Classes Challenge|Common Objects in Context|You Only look once|
|Developed by|University of Oxford|Microsoft|Joseph Redmon and Ali Farhadi|
|Number Classes|20|91|데이터에 따라 다름|
|Labelled Images|3000|2.5M|데이터에 따라 다름|

사물인식 알고리즘에 따라 인식표의 포맷이 전부 다름
Pascal VOC - XML
```
<annotation>
  <folder>images</folder>
  <filename>mak0.png</filename>
  <size>
    <width>512</width>
    <height>366</height>
    <depth>3</depth>
  </size>
  <segmented>0</segmented>
  <object>
    <name>without_mask</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <occluded>0</occluded>
    <bndbox>
      <xmin>79</xmin>
      <ymin>105</ymin>
      <xmax>109</xmax>
      <ymax>142</ymax>
    </bndbox>
  </object>
```
COCO - Json
```
{
"annotations": [
  {
  "segmentation": {
    "counts": [34, 55, 10, 71],
    "size": [240, 480]
  },
  "area": 600.4,
  "iscrowd": 1,
  "Image_id": 122214,
  "bbox": [473.05, 395.45, 38.65, 28.92],
  "category_id": 15,
  "id": 934
  }
  ]
}
```

YOLO
1 0.18 0.33 0.5 0.10
0 0.40 0.33 0.08 0.12


||Pascal VOC|COCO|YOLO|
|---|---|---|---|
|File type|XML|JSON|TXT|
|Nr.Files|One per image|one for complete dataset|one per image|
|Bounding Box|top-left point(xmin, ymin), bottom-right(xmax, ymax)|X top left, y top left, width, height|x center, y center, width, height|

