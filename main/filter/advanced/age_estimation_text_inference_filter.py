#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import numpy as np

from main.filter.basic.age_range_filter import AgeRangeFilter, MAX_AGE_VALUE, MAX_RANGE_POSSIBLE
from main.filter.filter import FILTERS_PROTO
from main.resource.text import Text
from main.tools.age_range import AgeRange

__author__ = "Ivan de Paz Centeno"


class AgeEstimationTextInferenceFilter(AgeRangeFilter):
    """
    Analyses text to extract an age range from it.
    """

    def __init__(self, weight, pattern, translate_dict=None, age_range_to_cover=None, max_age=MAX_AGE_VALUE, min_age=0,
                 max_range_distance_value=MAX_RANGE_POSSIBLE, strict_checks=False, capture_pattern_condition=None):
        """
        Constructor of the face detection filter
        :param weight: weight for this age estimation filter.
        :param pattern: pattern to match. Must extract a number or a [one-tweenty] text translatable to a number.
        :param age_range_to_cover: Age range object where estimated age_range should fit totally (with strict_checks)
            or partially.
        :param max_age: maximum value for detected age.
        :param min_age: minimum value for detected age
        :param max_range_distance_value: maximum range for the detected age
        :param strict_checks: if set to true, all the subfilters are strict. for example, with true, the age range
        estimated must fit inside the age_range_to_cover *completely*.
        :param capture_pattern_condition: Conditional for the capture pattern. Example of capture patterns:
                "== STRING" for capture pattern being an string match (case sensitive)
                "= STRING" for capture pattern being an string match (case insensitive)
        """
        self.compiled_regex = re.compile(pattern, re.IGNORECASE|re.MULTILINE)
        self.capture_pattern_condition = capture_pattern_condition

        if translate_dict is None:
            translate_dict = {}

        self.translate_dict = translate_dict

        AgeRangeFilter.__init__(self, weight, age_range_to_cover, max_age, min_age,
                 max_range_distance_value, strict_checks)

    def apply_to(self, text):
        """
        Applies this filter to the specified text.
        :param text: text object
        :return: True if filter passes. False otherwise.
        """
        translate_dict = self.translate_dict

        content = text.get_content()

        ages_strings = []

        matches = self.compiled_regex.findall(content)

        possible_ages = []
        for match in matches:
            age = match

            if age.lower() in translate_dict:
                possible_ages.append(translate_dict[age.lower()])
            else:
                try:
                    possible_ages.append(int(age))
                except Exception as ex:
                    ages_strings.append(age)

        if len(possible_ages) > 1:
            # We give more confidence to the attached text instead of the search query string.
            # The search query string is always the latest age of the list.
            # For this reason, we duplicate by 2 the number of possible ages below the last index.

            new_possible_ages = []

            for age in possible_ages:
                if age == possible_ages[-1]:
                    new_possible_ages += [age]
                else:
                    new_possible_ages += [age, age]

            possible_ages = new_possible_ages

        # We have a list of possible ages for this text inside possible_ages.
        reduced_ages = self.ivan_algorithm(possible_ages)

        if reduced_ages:

            min_age = int(min(reduced_ages))
            max_age = int(max(reduced_ages))

        else:
            min_age = -1
            max_age = -1

        return self._age_range_check_filter2(AgeRange(min_age, max_age), ages_strings)

    @staticmethod
    def ivan_algorithm(numbers_list, stdeviation=1):
        """
        Reduces the list numbers until their standard deviation is less than the specified value.
        :param numbers_list:
        :param stdeviation:
        :return: reduced version of the list.
        """
        numbers_list = list(numbers_list)
        standard_deviation = np.std(numbers_list)

        while standard_deviation > stdeviation:
            mean = np.mean(numbers_list)

            elements_in_max_side = sum(i > mean for i in numbers_list)
            elements_in_min_side = sum(i <= mean for i in numbers_list)

            if elements_in_max_side > elements_in_min_side:

                min_age = min(numbers_list)
                numbers_list.remove(min_age)
                min_age += standard_deviation
                numbers_list.append(min_age)

            elif elements_in_min_side > elements_in_max_side:

                max_age = max(numbers_list)
                numbers_list.remove(max_age)
                max_age -= standard_deviation
                numbers_list.append(max_age)

            else:

                max_age = max(numbers_list)
                min_age = min(numbers_list)
                numbers_list.remove(min_age)
                numbers_list.remove(max_age)
                min_age += standard_deviation
                max_age -= standard_deviation
                numbers_list.append(min_age)
                numbers_list.append(max_age)

            standard_deviation = np.std(numbers_list)

        return numbers_list

    def _age_range_check_filter2(self, age_range, captured_strings):
        """
        Checks initial filters against the specified age_range. If there's a capture pattern condition it will also be
        analyzed against it.
        :param age_range:
        :return: True if filter passes. False otherwise.
        """
        passed, self.weight, reason, age_range = AgeRangeFilter._age_range_check_filter(self, age_range)

        if passed:
            try:

                if self.capture_pattern_condition:

                    condition_statements = self.capture_pattern_condition.split(" ", 1)

                    if condition_statements[0] == "=":
                        condition_statements[1] = condition_statements[1].lower()

                    for text in captured_strings:

                        if condition_statements[0] == "=":
                            text = text.lower()

                        if text == condition_statements[1]:
                            break;

                    raise Exception("Capture pattern condition does not match any {} \"{}\".".format(
                        *condition_statements))


            except Exception as ex:
                reason = str(ex)
                passed = False

        return passed, self.weight, reason, age_range

    def get_type(self):
        return Text


FILTERS_PROTO[AgeEstimationTextInferenceFilter.__name__] = AgeEstimationTextInferenceFilter
