#!/usr/bin/env python
# -*- coding: utf-8 -*-
from main.filter.filter import Filter

__author__ = "Ivan de Paz Centeno"


MAX_RANGE_POSSIBLE = 9999999
MAX_AGE_VALUE = 9999999


class AgeRangeFilter(Filter):
    """
    Generic age range filter. Inherit this class to filter agerange-based estimations.
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
        Filter.__init__(self, weight)
        self.age_range_to_cover = age_range_to_cover
        self.max_age = max_age
        self.min_age = min_age

        self.max_range_distance_value = max_range_distance_value
        self.strict_checks = strict_checks

    def _age_range_check_filter(self, age_range):
        """
        Checks initial filters against the specified age_range.
        :return: True if filter passes. False otherwise.
        """

        try:
            if self.age_range_to_cover:
                age_range = age_range.intersect_with(self.age_range_to_cover)

            if self.strict_checks and age_range.get_distance() == 0:
                raise Exception("Computed age range does not fit in the specified range.")

            if age_range.get_distance() > self.max_range_distance_value:
                raise Exception("Range distance greater than desired.")

            if age_range.get_range()[0] < self.min_age:
                raise Exception("Minimum age not reached.")

            if age_range.get_range()[1] > self.max_age:
                raise Exception("Maximum age overpassed.")

            passed = True
            reason = ""

        except Exception as ex:
            reason = str(ex)
            passed = False

        return passed, self.weight, reason, age_range