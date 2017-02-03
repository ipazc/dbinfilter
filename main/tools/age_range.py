#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

__author__ = 'Iván de Paz Centeno'

INVALID_RANGE = -1


class AgeRange(object):
    """
    Represents a range of ages.
    It allows to perform basic computations between ranges, like intersection, distance or
    mean of the range.
    """

    def __init__(self, min_value, max_value):
        """
        Initializes the age range with the given range.
        """
        self.range = [min_value, max_value]

    @classmethod
    def from_string(cls, text):
        """
        Instantiates an age_range from a string text.


        :param text:  text representing the range. Format: "(x,y)" or "[x,y]"
                      Spaces will be stripped. Example: "(10, 50)"

        :return: instance of the age range with the specified range filled.
                 If the text is not valid or couldn't be parsed, the range will be
                 [INVALID_RANGE, INVALID_RANGE] and the return of is_valid() will be False.
        """

        if not text:
            text = ""

        text_components = text.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(",")

        if len(text_components) != 2:
            min_value = INVALID_RANGE
            max_value = INVALID_RANGE

        else:
            try:
                min_value = int(text_components[0].strip())
                max_value = int(text_components[1].strip())

            except Exception as error:
                min_value = INVALID_RANGE
                max_value = INVALID_RANGE

        return cls(min_value, max_value)

    def is_valid(self):
        """
        Checks if the range is a valid range or not.
        A valid range is greater than INVALID_RANGE in both sides of the range.
        :return:
        """
        return self.range[0] > -1 and self.range[1] > -1

    def get_range(self):
        """
        Getter for the range array
        :return: range array (2 components: min, max)
        """
        return self.range

    def get_distance(self):
        """
        Computes the distance between the max and the min of the range

        :return: distance between the max and the min of the range.
        """
        return self.range[1] - self.range[0]

    def get_mean(self):
        """
        Calculates the mean point of the range.

        :return: mean point of the range.
        """
        return int((self.range[1] + self.range[0]) / 2)

    def intersect_with(self, age_range):
        """
        Makes the intersection of this age range with the specified one.

        :param age_range:
        :return: The intersection age range object.
        """

        if age_range.get_range()[0] > self.get_range()[0]:
            intersection_min = age_range.get_range()[0]
        else:
            intersection_min = self.get_range()[0]

        if age_range.get_range()[1] < self.get_range()[1]:
            intersection_max = age_range.get_range()[1]
        else:
            intersection_max = self.get_range()[1]

        if intersection_min > intersection_max:
            intersection_min = -1
            intersection_max = -1

        return AgeRange(intersection_min, intersection_max)

    def to_dict(self):
        """
        :return: JSON-Compatible dictionary representation of the age.
        """
        return {'Age_range': "({}, {})".format(self.range[0], self.range[1])}

    def __str__(self):
        """
        :return: string representation of the age.
        """
        return json.dumps(self.to_dict())

    def hash(self):
        """
        Unique hash for same age ranges.
        :return:
        """
        return self.__str__()