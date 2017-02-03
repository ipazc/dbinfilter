#!/usr/bin/env python
# -*- coding: utf-8 -*-
from main.filter.filter import Filter

__author__ = "Ivan de Paz Centeno"


MAX_AREA_POSSIBLE = 9999999
MAX_DETECTIONS_POSSIBLE = 9999999


class BoundingBoxFilter(Filter):
    """
    Generic detection filter. Inherit this class to filter boundingbox-based detections.
    """

    def __init__(self, weight, should_detect=True, detection_location=None, min_detections=1,
                 max_detections = MAX_DETECTIONS_POSSIBLE, min_boundingbox_area=1,
                 max_boundingbox_area=MAX_AREA_POSSIBLE, strict_checks=False):
        """
        Constructor of the face detection filter
        :param weight: weight for this face detection filter.
        :param should_detect: Flag to indicate if the filter should filter by images of faces (if set to True)
            or by images of not faces (if set to False).
        :param detection_location: boundingbox object representing the area where the face should be detected to pass
            the filter.
        :param min_detections: number of faces to pass the filter.
        :param min_boundingbox_area: minimum area of the bounding boxes to pass the filter. If there's at least
        one bounding box not fitting this filter in the set
        :param max_boundingbox_area: maximum area of the bounding boxes to pass the filter.
        :param strict_checks: if set to true, all the subfilters are strict. for example, with true, ALL the
        bounding boxes must fit the specified boundingboxes area limits, instead of at least one.
        """
        Filter.__init__(self, weight)
        self.min_boundingbox_area = min_boundingbox_area
        self.max_boundingbox_area = max_boundingbox_area

        self.should_detect = should_detect
        self.detection_location = detection_location
        self.min_faces = min_detections
        self.max_faces = max_detections

        self.strict_checks = strict_checks

    def _bboxes_check_filter(self, bounding_boxes):
        """
        Checks initial filters against the specified bounding boxes.
        :return: True if filter passes. False otherwise.
        """

        try:

            original_bounding_boxes = bounding_boxes

            if self.detection_location:
                bounding_boxes = [bbox for bbox in bounding_boxes if
                                  bbox.intersect_with(self.detection_location).get_area() == bbox.get_area()]

            if self.strict_checks and len(original_bounding_boxes) < len(bounding_boxes):
                raise Exception("Not all the bounding boxes match the desired location")

            original_bounding_boxes = bounding_boxes

            bounding_boxes = [bbox for bbox in bounding_boxes if self.min_boundingbox_area <= bbox.get_area()
                              <= self.max_boundingbox_area]

            if self.strict_checks and len(original_bounding_boxes) < len(bounding_boxes):
                raise Exception("Not all the bounding boxes matches the desired size")

            if len(bounding_boxes) < self.min_faces:
                raise Exception("Not detected faces enough.")

            if len(bounding_boxes) > self.max_faces:
                raise Exception("Too much faces detected.")

            if self.should_detect and len(bounding_boxes) == 0:
                raise Exception("Faces detected when not desired.")

            passed = True
            reason = ""

        except Exception as ex:
            reason = str(ex)
            passed = False

        return passed, self.weight, reason, bounding_boxes