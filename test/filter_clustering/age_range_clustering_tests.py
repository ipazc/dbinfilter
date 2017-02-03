#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from main.filter_clustering.age_range_clustering import AgeRangeClustering
from main.tools.age_range import AgeRange


__author__ = "Ivan de Paz Centeno"


class AgeRangeClusteringTests(unittest.TestCase):
    """
    Test class for BoundingBoxClustering methods
    """

    def test_non_maximum_suppresion(self):
        """
        Tests if the NMS works correctly for age ranges.
        :return:
        """

        age_ranges = [
            AgeRange(5, 8),
            AgeRange(6, 9),
            AgeRange(10, 13),
        ]

        self.assertEqual(AgeRangeClustering._non_maximum_suppresion(age_ranges).get_range(), [5, 13])

    def test_filter_clustering(self):
        """
        Tests if clustering for ageranges filter works correctly.
        :return:
        """

        filter1_results = (True, 5, "", AgeRange(2, 5))
        filter2_results = (True, 3, "", AgeRange(3, 6))
        filter3_results = (True, 2, "", AgeRange(5, 7))
        filter4_results = (True, 1, "", AgeRange(6, 7))

        clustering = AgeRangeClustering([filter1_results, filter2_results, filter3_results, filter4_results])

        clustering.find_clusters()

        clusters = clustering.get_found_clusters()

        self.assertEqual(len(clusters), 2)

        postprocessed_clusters = {str(age_range.get_range()): weight for age_range, weight in clusters.items()}

        self.assertIn("[5, 7]", postprocessed_clusters)
        self.assertIn("[2, 6]", postprocessed_clusters)
        self.assertEqual(postprocessed_clusters["[5, 7]"], 3)
        self.assertEqual(postprocessed_clusters["[2, 6]"], 8)

if __name__ == '__main__':
    unittest.main()
