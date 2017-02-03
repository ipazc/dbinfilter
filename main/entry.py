#!/usr/bin/env python
# -*- coding: utf-8 -*-
from main.filter.advanced.age_estimation_filter import AgeEstimationFilter
from main.filter.advanced.face_detection_filter import FaceDetectionFilter
from main.filter.multifilter import Multifilter
from main.filter_clustering.age_range_clustering import AgeRangeClustering
from main.filter_clustering.bounding_box_clustering import BoundingBoxClustering
from main.resource.image import Image

__author__ = "Ivan de Paz Centeno"

BACKEND = 'http://192.168.2.110:9095'
MAX_FACES = 1

TYPE_MAP = {
    "face-detection": "detection-requests/faces",
    "face-age-estimation": "estimation-requests/age/face",
}

def build_api_url(type, method="stream", service_name="default"):
    return "{}/{}/{}?service={}".format(BACKEND, TYPE_MAP[type], method, service_name)


image = Image(uri='/var/www/datasets/test_1484735758.51682/google/8.jpe')
image.load_from_uri()

face_filters = [
    FaceDetectionFilter(2, build_api_url("face-detection", service_name="opencv-cpu-haarcascade-face-detection"), max_faces=MAX_FACES),
    FaceDetectionFilter(5, build_api_url("face-detection", service_name="dlib-cpu-hog-svm-face-detection"), max_faces=MAX_FACES),
    FaceDetectionFilter(8, build_api_url("face-detection", service_name="mt-gpu-caffe-cnn-face-detection"), max_faces=MAX_FACES),
]

multifilter = Multifilter(face_filters)
face_scores =  multifilter.apply_to(image)

bbox_clustering = BoundingBoxClustering(face_scores)

bbox_clustering.find_clusters()
bounding_box_clusters = bbox_clustering.get_found_clusters()

detection_weight_threshold = int(sum([weight for (_,weight,_,_) in face_scores]) / 2)

# Since each image should have only one face, the cluster list should have only one valid group.
valid_boxes = [bounding_box for bounding_box, weight in bounding_box_clusters.items() if
               weight >= detection_weight_threshold]

if len(valid_boxes) > MAX_FACES:
    print("The image is probably composed by more than {} faces. Discarded.".format(MAX_FACES))
    exit(-1)

for bounding_box in valid_boxes:
    bounding_box.expand(0.4)
    bounding_box.fit_in_size(image.get_size())

print([str(bounding_box) for bounding_box in valid_boxes])

cropped_images = [image.crop_image(bounding_box, "cropped_face") for bounding_box in valid_boxes]

age_filters = [
    AgeEstimationFilter(12, build_api_url("face-age-estimation", service_name="gpu-cnn-rothe-real-age-estimation"), min_age=0, max_age=99),
    AgeEstimationFilter(10, build_api_url("face-age-estimation", service_name="gpu-cnn-rothe-apparent-age-estimation"), min_age=0, max_age=99),
    AgeEstimationFilter(4, build_api_url("face-age-estimation", service_name="gpu-cnn-levi-hassner-age-estimation"), min_age=0, max_age=99),
]

multifilter = Multifilter(age_filters)
face_age_scores = multifilter.apply_to_list(cropped_images)

print("Total score: {}".format(sum ([result * weight for (result, weight, reason, boxes) in face_age_scores])))
[print(age) for (result, weight, reason, age) in face_age_scores if result]


detection_weight_threshold = int(sum([weight for (_,weight,_,_) in face_age_scores]) / 2.2)

age_clustering = AgeRangeClustering(face_age_scores)

age_clustering.find_clusters()
age_range_clusters = age_clustering.get_found_clusters()

# Since each face should have only one age_range, the cluster list should have only one valid group.
valid_ages = {age_range: weight for age_range, weight in age_range_clusters.items() if
               weight >= detection_weight_threshold}

#api_url =

#filter_age =

#result, weight, reason, age_range = filter_age.appy_to(crop_image)

print(["{}: {} (from {} threshold)".format(age_range.get_range(), weight, detection_weight_threshold) for age_range, weight in valid_ages.items()])
