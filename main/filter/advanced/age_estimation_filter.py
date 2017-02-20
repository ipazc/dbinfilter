#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

import requests

from main.filter.basic.age_range_filter import AgeRangeFilter, MAX_AGE_VALUE, MAX_RANGE_POSSIBLE
from main.filter.filter import FILTERS_PROTO
from main.resource.image import Image
from main.tools.age_range import AgeRange

__author__ = "Ivan de Paz Centeno"


class AgeEstimationFilter(AgeRangeFilter):
    """
    Applies an age estimation filter to an image.
    """

    def __init__(self, weight, api_url, age_range_to_cover=None, max_age=MAX_AGE_VALUE, min_age=0,
                 max_range_distance_value=MAX_RANGE_POSSIBLE, strict_checks=False):
        """
        Constructor of the age estimation filter.
        :param weight: weight for this face detection filter.
        :param api_url: URL to CVMLModulerized age estimator.
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
                                                                                        AgeEstimationFilter.__name__))

        response_json = json.loads(response.text)
        if 'Age_range' not in response_json:
            raise Exception("This filter does not understand backend's language. It may be a different version.")

        age_range = AgeRange.from_string(response_json['Age_range'])

        return self._age_range_check_filter(age_range)

    def get_type(self):
        return Image


FILTERS_PROTO[AgeEstimationFilter.__name__] = AgeEstimationFilter
