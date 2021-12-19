#!/usr/bin/env python
# coding: utf-8

# # Train YOLO object detector with Turi Create


import os, sys, math
import pandas as pd
import turicreate as tc


# Helper code for loading the CSV file and combining it with an SFrame. We only keep the images that we have annotations for.


def row_to_annotations(row, img_width, img_height):
    x_min = row['x_min']
    x_max = row['x_max']
    y_min = row['y_min']
    y_max = row['y_max']
    h = int((y_max - y_min) * img_height)
    w = int((x_max - x_min) * img_width)
    x = int((x_max + x_min) * img_width / 2)
    y = int((y_max + y_min) * img_height / 2)
    return {
        "coordinates": {"height":h, 'width':w, 'x':x, 'y':y},
        'label': row['class_name']
    }

def load_images_with_annotations(images_dir, annotations_file):
    # Load the images into a Turi SFrame.
    data = tc.image_analysis.load_images(images_dir, with_path=True)
    
    # Load the annotations CSV file into a Pandas dataframe.
    csv = pd.read_csv(annotations_file)

    # Loop through all the images and match these to the annotations from the
    # CSV file, if annotations are available for the image.
    all_annotations = []
    for i, item in enumerate(data):
        # Grab image info from the SFrame.
        img_path = item["path"]
        img_width = item["image"].width
        img_height = item["image"].height

        # Find the corresponding row(s) in the CSV's dataframe.
        image_id = os.path.basename(img_path)[:-4]
        rows = csv[csv["image_id"] == image_id]
        #print(rows)
        # Turi expects a list for every image that contains a dictionary for
        # every bounding box that we have an annotation for.
        img_annotations = []
        if len(rows):
            img_annotations = rows.apply(
                lambda x:row_to_annotations(x, img_height=img_height, img_width=img_width),
                axis = 1
            ).to_list()
          
        #print(img_annotations)
        # The CSV file stores the coordinate as numbers between 0 and 1,
        # but Turi wants pixel coordinates in the image.
        
        # A bounding box in Turi is given by a center coordinate and the
        # width and height, we have them as the four corners of the box.

            
            # img_annotations.append({"coordinates": {"height": height, 
            #                                         "width": width, 
            #                                         "x": x, 
            #                                         "y": y}, 
            #                         "label": class_name})

        # If there were no annotations for this image, then append a None
        # so that we can filter out those images from the SFrame.
        if len(img_annotations) > 0:
            all_annotations.append(img_annotations)
        else:
            all_annotations.append(None)

    data["annotations"] = tc.SArray(data=all_annotations, dtype=list)
    return data.dropna()



data_dir = "snacks"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")

train_data = load_images_with_annotations(train_dir, data_dir + "/annotations-train.csv")

train_data['image_with_ground_truth'] = tc.object_detector.util.draw_bounding_boxes(
                                            train_data['image'], train_data['annotations'])
model = tc.object_detector.create(train_data, feature='image', annotations='annotations')

model.save("SnackDetector.model")

model.export_coreml("SnackDetector.mlmodel")

model = tc.load_model("SnackDetector.model")

val_data = load_images_with_annotations(val_dir, data_dir + "/annotations-val.csv")
test_data = load_images_with_annotations(test_dir, data_dir + "/annotations-test.csv")


# Make predictions on the test data. This outputs something like this:
# 
# ```
# [{'confidence': 0.7225357099539148,
#   'coordinates': {'height': 73.92794444010806,
#                   'width': 90.45315889211807,
#                   'x': 262.2198759929745,
#                   'y': 155.496952970812},
#   'label': 'dog',
#   'type': 'rectangle'},
#  ...]
# ```
# 
# which is similar to the annotations, but now there is a `confidence` field as well.