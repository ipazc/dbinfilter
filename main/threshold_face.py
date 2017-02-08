#!/usr/bin/env python
# -*- coding: utf-8 -*-
from main.filter.advanced.face_detection_filter import FaceDetectionFilter
from main.filter.multifilter import Multifilter
from main.filter_clustering.bounding_box_clustering import BoundingBoxClustering
from main.resource.image import Image
from main.tools.boundingbox import BoundingBox

__author__ = "Ivan de Paz Centeno"

BACKEND = 'http://192.168.2.110:9095'
MAX_FACES = 15

TYPE_MAP = {
    "face-detection": "detection-requests/faces",
    "face-age-estimation": "estimation-requests/age/face",
}

def build_api_url(type, method="stream", service_name="default"):
    return "{}/{}/{}?service={}".format(BACKEND, TYPE_MAP[type], method, service_name)


gt = [
    Image(uri="/home/ivan/Documents/threshold_test/1.jpg", metadata=[BoundingBox(548, 16,1100, 1100)]),
    Image(uri="/home/ivan/Documents/threshold_test/2.jpg", metadata=[BoundingBox(64, 55, 40, 40),
                                                                      BoundingBox(109, 41, 35, 41),
                                                                      BoundingBox(157, 56, 35, 37),
                                                                      BoundingBox(211, 37, 35, 35),
                                                                      BoundingBox(251, 58, 31, 31),
                                                                      BoundingBox(290, 50, 40, 40),
                                                                      BoundingBox(335, 65, 31, 35),
                                                                      BoundingBox(381, 49, 32, 34)]),
    Image(uri="/home/ivan/Documents/threshold_test/3.jpg", metadata=[BoundingBox(280, 16, 206, 245)]),
    Image(uri="/home/ivan/Documents/threshold_test/4.jpg", metadata=[BoundingBox(18, 130, 300, 276)]),
    Image(uri="/home/ivan/Documents/threshold_test/5.jpg", metadata=[BoundingBox(180, 80, 224, 235)]),
    Image(uri="/home/ivan/Documents/threshold_test/6.jpg", metadata=[BoundingBox(65, 24, 180, 173)]),
    Image(uri="/home/ivan/Documents/threshold_test/7.jpg", metadata=[BoundingBox(57, 17, 80, 95)]),
    Image(uri="/home/ivan/Documents/threshold_test/8.jpg", metadata=[BoundingBox(90, 26, 240, 250)]),
    Image(uri="/home/ivan/Documents/threshold_test/9.jpg", metadata=[BoundingBox(125, 64, 120, 140)]),
    Image(uri="/home/ivan/Documents/threshold_test/10.jpg", metadata=[BoundingBox(200, 185, 250, 245)]),
]

for image in gt: image.load_from_uri()


face_filters = [
    FaceDetectionFilter(1, build_api_url("face-detection", service_name="opencv-cpu-haarcascade-face-detection"), max_faces=MAX_FACES),
    FaceDetectionFilter(100, build_api_url("face-detection", service_name="dlib-cpu-hog-svm-face-detection"), max_faces=MAX_FACES),
    FaceDetectionFilter(10000, build_api_url("face-detection", service_name="mt-gpu-caffe-cnn-face-detection"), max_faces=MAX_FACES),
]

face_scores_set_to_add = [[(True, 1000000, "", [bounding_box for bounding_box in image.get_metadata()])] for image in gt]

multifilter = Multifilter(face_filters)
face_scores_set = [multifilter.apply_to(image) + bias_score for image, bias_score in zip(gt, face_scores_set_to_add)]

bbox_clustering_list = [BoundingBoxClustering(scores, cluster_match_rate=0.7) for scores in face_scores_set]

[bbox_clustering.find_clusters() for bbox_clustering in bbox_clustering_list]

bounding_box_clusters_list = [bbox_clustering.get_found_clusters() for bbox_clustering in bbox_clustering_list]

index = 1
filters_scores = {filter.get_weight():0 for filter in face_filters}
filters_scores[1000000] = 0

def decompose(weight, operands):
    decomposition = {operand: 0 for operand in operands}
    operands = list(operands)

    while weight > 0 and len(operands) > 0:

        max_operand = max(operands)

        if weight >= max_operand:
            decomposition[max_operand] += 1
            weight -= max_operand

        else:
            operands.remove(max_operand)

    return decomposition

for clusters in bounding_box_clusters_list:
    print("For image {}".format(index))
    [print(bounding_box, weight) for bounding_box, weight in clusters.items()]
    index += 1

    for _, weight in clusters.items():

        decomposition = decompose(weight, list(filters_scores.keys()))
        print("{} decomposed in {}".format(weight, decomposition))

        clustered_well = decomposition[max(list(decomposition.keys()))] == 1
        for key, value in decomposition.items():

            if value > 1 or not clustered_well:
                filters_scores[key] -= (value - 1)
            elif value == 1 and clustered_well:
                filters_scores[key] += 1

print(filters_scores)

# Lets normalize the thresholds calculated
max_count = max(filters_scores.values())
del filters_scores[max(list(filters_scores.keys()))]

filters_scores = {key: round(value/max_count, 2) for key, value in filters_scores.items()}
max_sum = sum(list(filters_scores.values()))
filters_scores = {key: round(value/max_sum, 2) for key, value in filters_scores.items()}
print(filters_scores)

sorted_filters = filters_scores.values()

threshold = sorted_filters

# Since each image should have only one face, the cluster list should have only one valid group.
#valid_boxes = [bounding_box for bounding_box, weight in bounding_box_clusters.items() if
#               weight >= detection_weight_threshold]
