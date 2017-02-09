#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

import shutil

from main.dataset.generic_image_age_dataset import GenericImageAgeDataset
from main.dataset.raw_crawled_dataset import RawCrawledDataset
from main.filter.advanced.age_estimation_filter import AgeEstimationFilter
from main.filter.advanced.age_estimation_text_inference_filter import AgeEstimationTextInferenceFilter
from main.filter.advanced.face_detection_filter import FaceDetectionFilter
from main.filter.multifilter import Multifilter
from main.filter_clustering.age_range_clustering import AgeRangeClustering
from main.filter_clustering.bounding_box_clustering import BoundingBoxClustering
from main.resource.image import Image
from main.resource.text import Text
from main.tools.age_range import AgeRange

__author__ = "Ivan de Paz Centeno"

SAVE_BATCH_AMMOUNT = 200
MAX_IMAGE_SIZE = (1200, 1200)

if len(sys.argv) != 2:
    print("A parameter for folder/zip location of the processable dataset is needed.")
    exit(-1)

source = sys.argv[1]

if not os.path.exists(source):
    print("Given folder/filename does not exist.")
    exit(-1)

if os.path.isdir(source): source_type = "DIR"
else: source_type = "FILE"


BACKEND = 'http://192.168.2.110:9095'
#MAX_FACES = 1

TYPE_MAP = {
    "face-detection": "detection-requests/faces",
    "face-age-estimation": "estimation-requests/age/face",
}

def build_api_url(type, method="stream", service_name="default"):
    return "{}/{}/{}?service={}".format(BACKEND, TYPE_MAP[type], method, service_name)

def create_temp_folder(name="GUID"):
    """
    Creates a temporal folder for dataset process
    :param name: name to assign.
    :return:
    """

    if name == "GUID":
        import uuid
        name = uuid.uuid4()

    if not os.path.exists("/tmp/inferencedb/"):
        os.mkdir("/tmp/inferencedb/")

    folder = "/tmp/inferencedb/{}".format(name)

    os.mkdir(folder)

    return folder


print("Importing dataset from {} ({})".format(source, source_type))


if source_type == "DIR":
    raw_dataset = RawCrawledDataset(source)
    folder = source

else:
    folder = create_temp_folder()

    print("Temporal folder on \"{}\"".format(folder))
    raw_dataset = RawCrawledDataset(folder)
    raw_dataset.import_from_zip(source)

new_folder = create_temp_folder()
age_dataset = GenericImageAgeDataset(new_folder)

try:
    raw_dataset.load_dataset()

    if len(raw_dataset.get_metadata_content()) == 0:
        print("Dataset source seems empty or not understandable by the inference filter. Aborted.")
        print("Ensure that there exists a metadata.json file in the same folder.")
        exit(-1)

    print("Detected {} elements in raw dataset.".format(len(raw_dataset.get_metadata_content())))


    # Let's define the filters:

    face_filter = FaceDetectionFilter(1, build_api_url("face-detection", service_name="mt-gpu-caffe-cnn-face-detection"), min_faces=1)

    image_age_filters = [
        #AgeEstimationFilter(2, build_api_url("face-age-estimation",
        #                    service_name="gpu-cnn-rothe-real-age-estimation"), min_age=0,
        #                    max_age=99),
        #AgeEstimationFilter(2, build_api_url("face-age-estimation",
        #                                     service_name="gpu-cnn-rothe-apparent-age-estimation"), min_age=0,
        #                    max_age=99),
        AgeEstimationFilter(4, build_api_url("face-age-estimation",
                                             service_name="gpu-cnn-levi-hassner-age-estimation"), min_age=0,
                            max_age=99),
    ]

    translate_dict1 = {
        "zero": 0,
        "oh": 0,
        "zip": 0,
        "zilch": 0,
        "nada": 0,
        "bupkis": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
    }

    translate_dict2 = {
        "fir":1,
        "second":2,
        "thi":3,
        "four":4,
        "five":5,
        "six":6,
        "seven":7,
        "eight":8,
        "nine":9,
        "ten":10,
        "eleven":11,
        "twelve":12,
        "thirteen":13,
        "forteen":14,
        "fifteen":15,
        "sixteen":16,
        "seventeen":17,
        "eighteen":18,
        "nineteen":19,
        "twent":20
    }

    text_age_filters = [
        AgeEstimationTextInferenceFilter(5,
                                         pattern="(?:<b>)?([0-9][0-9]?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|forteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty).?(?:<\/b>)?(?:[ ]|[-])(?:<b>)?(?:(?:year[']?s?)(?:(?:<\/b>)?(?:[ ]|[-])(?:<b>)?(?:old)(?:<\/b>)?)|(?:yo))",
                                         translate_dict=translate_dict1),
        AgeEstimationTextInferenceFilter(6,
                                         pattern="(?:baby).*(?:<b>)?([0-9][0-9]?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|forteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty).?(?:<\/b>)?(?:[ ]|[-])(?:<b>)?(?:(?:year[']?s?)(?:(?:<\/b>)?(?:[ ]|[-])(?:<b>)?(?:old)(?:<\/b>)?)|(?:yo))",
                                         translate_dict=translate_dict1, max_age=3),
        AgeEstimationTextInferenceFilter(6,
                                         pattern="(?:<b>)?([0-9][0-9]?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|forteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty).?(?:<\/b>)?(?:[ ]|[-])(?:<b>)?(?:(?:year[']?s?)(?:(?:<\/b>)?(?:[ ]|[-])(?:<b>)?(?:old)(?:<\/b>)?)|(?:yo)).*(?:baby)",
                                         translate_dict=translate_dict1, max_age=3),
        AgeEstimationTextInferenceFilter(7,
                                         pattern="(?:<b>)?([0-9][0-9]?|fir|second|thi|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|forteen|fifteen|sixteen|seventeen|eighteen|nineteen|twent)(?:st|nd|rd|th|ieth)?(?:<\/b>)?(?:[ ]|[-])(?:<b>)?(?:birthday)(?:<\/b>)?",
                                         translate_dict=translate_dict2)
    ]

    image_multifilter = Multifilter(image_age_filters)
    text_multifilter = Multifilter(text_age_filters)
    # Let's process each image.
    metadata_content = raw_dataset.get_metadata_content()

    iteration = 0
    size_metadata = len(metadata_content)
    count = 0
    for image_hash, data in metadata_content.items():
        count +=1
        if not 'metadata' in data:

            print("Element {} hierarchy in the metadata is not correct (does not contain metadata for the image's hash "
                  "key). May it be a different version of dataset? skipped.".format(image_hash))
            continue

        metadata = data['metadata']
        text = Text(content=metadata['desc'])

        if not 'uri' in metadata:

            print("Element's {} metadata does not reference any URI. May it be a different version of dataset? "
                  "skipped.".format(image_hash))
            continue

        uri = os.path.join(folder, metadata['uri'][0])
        print("loading {}".format(uri))
        image = Image(uri)
        image.load_from_uri()

        if not image.is_loaded():
            continue

        if image.get_size() > MAX_IMAGE_SIZE:
            image.resize_to(MAX_IMAGE_SIZE)

        (faces_detected, weight, reason, bounding_boxes) = face_filter.apply_to(image)

        if not faces_detected:
            print("No faces detected for file {} ({})".format(image_hash, uri))
            continue

        print("Detected {} faces in {} ({}): \n{}".format(len(bounding_boxes), image_hash, uri, "\n".join([str(bounding_box) for bounding_box in bounding_boxes])))

        for bounding_box in bounding_boxes:

            bounding_box.expand(0.2)
            bounding_box.fit_in_size(image.get_size())
            new_image = image.crop_image(bounding_box, new_uri="None")

            print(metadata['desc'])
            # Let's inference the age.
            desc = metadata['desc'].split(';')[-1]

            try:
                age = int(desc[0])
            except Exception as ex:
                age = 0

            face_age_scores = text_multifilter.apply_to(text) + image_multifilter.apply_to(new_image) #+ text_multifilter.apply_to(text)

            # Let's discard all those scores that didn't pass the filter.
            face_age_scores = [(result, weight, reason, age) for (result, weight, reason, age) in face_age_scores if result]

            [print(age, "x", weight) for (result, weight, reason, age) in face_age_scores if result]
            # Now we need to map the scores into a list.
            ages_list = []
            for (result, weight, reason, age) in face_age_scores:
                ages_list += age.get_range() * weight

            reduced_list = AgeEstimationTextInferenceFilter.ivan_algorithm(ages_list)

            age_range = AgeRange(int(min(reduced_list)), int(max(reduced_list)))

            print("Inferred age: {}".format(age_range))

            new_image.metadata=[age_range]
            age_dataset.put_image(new_image)

            if iteration % SAVE_BATCH_AMMOUNT == 0:
                print("[{}%] Saved dataset into \"{}\".".format(round(count/size_metadata * 100, 2), new_folder))
                age_dataset.save_dataset()

            iteration += 1

finally:
    print("Saved dataset into \"{}\".".format(new_folder))
    age_dataset.save_dataset()

    if source_type=="FILE":
        print("Removing temporary folder {}...".format(folder))
        shutil.rmtree(folder)
        print("Done.")

    #shutil.rmtree(new_folder)


"""face_filters = [
    FaceDetectionFilter(2, build_api_url("face-detection", service_name="opencv-cpu-haarcascade-face-detection")),
    FaceDetectionFilter(5, build_api_url("face-detection", service_name="dlib-cpu-hog-svm-face-detection")),
    FaceDetectionFilter(8, build_api_url("face-detection", service_name="mt-gpu-caffe-cnn-face-detection")),
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

#print(["{}: {} (from {} threshold)".format(age_range.get_range(), weight, detection_weight_threshold) for age_range, weight in valid_ages.items()])
"""