#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from main.filter.advanced.age_estimation_text_inference_filter import AgeEstimationTextInferenceFilter
from main.resource.text import Text


__author__ = "Ivan de Paz Centeno"


class AgeEstimationTextInferenceFilterTests(unittest.TestCase):
    """
    Test class for AgeEstimationTextINferenceFIlter methods
    """

    def test_age_range_by_text_pattern_yearold(self):
        """
        Tests if the age range filtered from a given text is correct or not.
        :return:
        """
        text = Text(content="Cute 6 <b>Year</b> <b>Old</b> <b>Boy</b> With Blonde Hair Seven <b>year</b> <b>old</b> stock photos ...;1 year old boy;")
        pattern = "(?:<b>)?([0-9][0-9]?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|forteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty).?(?:<\/b>)?(?:[ ]|[-])(?:<b>)?(?:(?:year[']?s?)(?:(?:<\/b>)?(?:[ ]|[-])(?:<b>)?(?:old)(?:<\/b>)?)|(?:yo))"
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

        filter = AgeEstimationTextInferenceFilter(5, pattern, translate_dict=translate_dict1)

        (result, weight, reason, age_range) = filter.apply_to(text)

        self.assertTrue(result)
        self.assertEqual(age_range.get_range(), [4, 7])

        text2 = Text(content="Cute Black 12 <b>Year</b> <b>Old</b> <b>Boys</b> Street stylin&#39; style squad ackermans;1 year old boy;")

        (result, weight, reason, age_range) = filter.apply_to(text2)

        self.assertTrue(result)
        self.assertEqual(age_range.get_range(), [10, 12])

if __name__ == '__main__':
    unittest.main()
