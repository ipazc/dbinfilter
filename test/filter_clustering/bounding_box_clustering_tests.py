#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from main.filter_clustering.bounding_box_clustering import BoundingBoxClustering
from main.tools.boundingbox import BoundingBox

__author__ = "Ivan de Paz Centeno"


class BoundingBoxClusteringTests(unittest.TestCase):
    """
    Test class for BoundingBoxClustering methods
    """

    def test_non_maximum_suppresion(self):
        """
        Tests if the NMS works correctly for boundingboxes.
        :return:
        """

        boxes = [
            BoundingBox(2, 3, 3, 3),
            BoundingBox(1, 4, 3, 3),
        ]

        self.assertEqual(BoundingBoxClustering._non_maximum_suppresion(boxes).get_box(), [1, 3, 4, 4])

    def test_filter_clustering(self):
        """
        Tests if clustering for boundingboxes filter works correctly.
        :return:
        """

        filter1_results = (True, 5, "", [BoundingBox(2, 3, 3, 3)])
        filter2_results = (True, 3, "", [BoundingBox(1, 4, 3, 3)])
        filter3_results = (True, 1, "", [BoundingBox(5, 5, 3, 3)])

        clustering = BoundingBoxClustering([filter1_results, filter2_results, filter3_results])

        clustering.find_clusters()

        clusters = clustering.get_found_clusters()

        self.assertEqual(len(clusters), 2)

        postprocessed_clusters = {str(bbox.get_box()): weight for bbox, weight in clusters.items()}

        self.assertIn("[5, 5, 3, 3]", postprocessed_clusters)
        self.assertIn("[1, 3, 4, 4]", postprocessed_clusters)
        self.assertEqual(postprocessed_clusters["[1, 3, 4, 4]"], 8)
        self.assertEqual(postprocessed_clusters["[5, 5, 3, 3]"], 1)


if __name__ == '__main__':
    unittest.main()
