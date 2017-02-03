#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json

import requests

from main.filter.basic.bounding_box_filter import BoundingBoxFilter, MAX_DETECTIONS_POSSIBLE, MAX_AREA_POSSIBLE
from main.filter.filter import FILTERS_PROTO
from main.resource.image import Image
from main.tools.boundingbox import BoundingBox

__author__ = "Ivan de Paz Centeno"



class FaceDetectionFilter(BoundingBoxFilter):
    """
    Applies a face detection filter to an image.
    """

    def __init__(self, weight, api_url, should_detect_face=True, face_location=None, min_faces=1,
                 max_faces=MAX_DETECTIONS_POSSIBLE, min_boundingbox_area=1, max_boundingbox_area=MAX_AREA_POSSIBLE,
                 strict_checks=False):
        """
        Constructor of the face detection filter
        :param weight: weight for this face detection filter.
        :param api_url: URL to CVMLModulerized face detector.
        :param should_detect_face: Flag to indicate if the filter should filter by images of faces (if set to True)
            or by images of not faces (if set to False).
        :param face_location: boundingbox object representing the area where the face should be detected to pass
            the filter.
        :param min_faces: number of faces to pass the filter.
        :param min_boundingbox_area: minimum area of the bounding boxes to pass the filter. If there's at least one
        :param max_boundingbox_area: maximum area of the bounding boxes to pass the filter.
        """
        BoundingBoxFilter.__init__(self, weight, should_detect_face, face_location, min_faces, max_faces,
                                   min_boundingbox_area, max_boundingbox_area, strict_checks)

        self.api_url = api_url

    def apply_to(self, image):
        """
        Applies this filter to the specified image.
        :param image:
        :return: True if filter passes. False otherwise.
        """

        response = requests.put(self.api_url, data=image.get_jpeg())

        if response.status_code != 200:
            raise Exception("Backend ({}) for filtering with {} is returning a bad response!".format(self.api_url,
                                                                                        FaceDetectionFilter.__name__))

        response_json = json.loads(response.text)

        if 'bounding_boxes' not in response_json:
            raise Exception("This filter does not understand backend language. It may be a different version.")

        bounding_boxes = [ BoundingBox.from_string(bbox_string)
                           for bbox_string in response_json['bounding_boxes'] ]

        return self._bboxes_check_filter(bounding_boxes)

    def get_type(self):
        return Image


FILTERS_PROTO[FaceDetectionFilter.__name__] = FaceDetectionFilter
