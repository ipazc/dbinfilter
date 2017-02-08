#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

import requests

from main.filter.basic.age_range_filter import AgeRangeFilter, MAX_AGE_VALUE, MAX_RANGE_POSSIBLE
from main.filter.filter import FILTERS_PROTO
from main.resource.image import Image
from main.tools.age_range import AgeRange

__author__ = "Ivan de Paz Centeno"


class AgeEstimationTextInferenceFilter(AgeRangeFilter):
    """
    Analyses text to extract an age range from it.
    """

    def __init__(self, weight, age_range_to_cover=None, max_age=MAX_AGE_VALUE, min_age=0,
                 max_range_distance_value=MAX_RANGE_POSSIBLE, strict_checks=False):
        """
        Constructor of the face detection filter
        :param weight: weight for this face detection filter.
        :param age_range_to_cover: Age range object where estimated age_range should fit totally (with strict_checks)
            or partially.
        :param max_age: maximum value for detected age.
        :param min_age: minimum value for detected age
        :param max_range_distance_value: maximum range for the detected age
        :param strict_checks: if set to true, all the subfilters are strict. for example, with true, the age range
        estimated must fit inside the age_range_to_cover *completely*.
        """
        AgeRangeFilter.__init__(self, weight, age_range_to_cover, max_age, min_age,
                 max_range_distance_value, strict_checks)

    def apply_to(self, text):
        """
        Applies this filter to the specified text.
        :param text: text object
        :return: True if filter passes. False otherwise.
        """


        return self._age_range_check_filter(age_range)

    def get_type(self):
        return Image


FILTERS_PROTO[AgeEstimationTextInferenceFilter.__name__] = AgeEstimationTextInferenceFilter
