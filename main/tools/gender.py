#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

__author__ = 'Iván de Paz Centeno'

GENDER_MALE = 0
GENDER_FEMALE = 1
GENDER_UNKNOWN = 2


class Gender(object):
    """
    Represents a gender (male or female). Provides basic functionality like checks
    if a string is a valid gender or not
    """

    def __init__(self, gender_value):
        """
        Initializes the gender with the specified value
        """
        self.gender_value = gender_value

    def get_gender(self):
        """
        Getter for the gender int representation
        :return: Integer representation of the gender (GENDER_MALE, GENDER_FEMALE or GENDER_UNKNOWN)
        """
        return self.gender_value

    def to_dict(self):
        """
        :return: JSON-Compatible dictionary representation of the age.
        """
        result = "Unknown"

        if self.gender_value == GENDER_MALE:
            result = "Male"
        elif self.gender_value == GENDER_FEMALE:
            result = "Female"

        return {'Gender': result}

    def __str__(self):
        """
        :return: string representation of the age.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_string(cls, gender_string):
        """
        Instantiates a gender from a text string
        :param gender_string: valid string for a gender.
            * "f", "female", in Upper/lower case for females
            * "m", "male", in Upper/lower case for males.

        :return: object representing the gender.
        """

        if not gender_string:
            gender_string = ""

        gender_to_return = GENDER_UNKNOWN
        gender_string = gender_string.lower()

        if gender_string=='f' or gender_string=='female':
            gender_to_return = GENDER_FEMALE
        elif gender_string=='m' or gender_string=='male':
            gender_to_return = GENDER_MALE

        return cls(gender_to_return)

    def is_valid(self):
        """
        Checks whether the gender is a valid gender or not.
        :return:  True if gender is valid (non GENDER_UNKNOWN). False otherwise.
        """
        return self.gender_value != GENDER_UNKNOWN