#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from main.filter_clustering.filter_composition_clustering import FilterCompositionClustering
from main.tools.boundingbox import BoundingBox

__author__ = "Ivan de Paz Centeno"


class BoundingBoxClustering(FilterCompositionClustering):
    """
    Clusterizes filters of bounding boxes.
    """

    def find_clusters(self):
        """
        Finds the clusters for the wrapped filters results
        :return:
        """

        # Let's inflate the clusters dictionary to link bounding boxes inside.
        clusters = {cluster_box: {'weight': weight, 'bboxes': []} for cluster_box, weight in self.clusters.items()}

        for (result, weight, _, bounding_boxes) in self.filter_result_list:

            if not result:
                continue

            # Are there any cluster in the list to match the boxes?
            for bounding_box in bounding_boxes:

                clustered = False

                for cluster_box in clusters:

                    intersection_rate = self._get_intersect_rate(cluster_box, bounding_box)

                    if intersection_rate >= self.cluster_match_rate:
                        clusters[cluster_box]['weight'] += weight
                        clusters[cluster_box]['bboxes'].append(bounding_box)
                        clustered = True
                        break

                if not clustered:
                    clusters[bounding_box] = {'weight': weight, 'bboxes': []}

        # Now we deflate the clusters by calculating the NMS of the bounding boxes linked together.
        self.clusters = {self._non_maximum_suppresion([cluster_box]+content['bboxes']): content['weight']
                         for cluster_box, content in clusters.items()}

    @staticmethod
    def _non_maximum_suppresion(bounding_boxes):
        """
        Computes the non maximum suppression of a list of bounding boxes.
        :param bounding_boxes:
        :return:
        """
        minimum_x = min([bbox.get_x() for bbox in bounding_boxes])
        minimum_y = min([bbox.get_y() for bbox in bounding_boxes])

        maximum_x = max([bbox.get_x() + bbox.get_width() for bbox in bounding_boxes])
        maximum_y = max([bbox.get_y() + bbox.get_height() for bbox in bounding_boxes])

        width = maximum_x - minimum_x
        height = maximum_y - minimum_y

        return BoundingBox(minimum_x, minimum_y, width, height)

    def _get_intersect_rate(self, boundingbox1, boundingbox2):
        """
        Computes the intersection rate for both bounding boxes.
        :param boundingbox1: box to intersect from
        :param boundingbox2: box to intersect with
        :return: percent of intersection with the smallest bounding box
        """

        min_area = min([boundingbox1.get_area(), boundingbox2.get_area(), 1])

        intersection = boundingbox1.intersect_with(boundingbox2)

        return intersection.get_area() / min_area