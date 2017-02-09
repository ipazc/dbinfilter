#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from main.filter_clustering.filter_composition_clustering import FilterCompositionClustering
from main.tools.age_range import AgeRange

__author__ = "Ivan de Paz Centeno"


class AgeRangeClustering(FilterCompositionClustering):
    """
    Clusterizes filters of age range.
    """

    def find_clusters(self):
        """
        Finds the clusters for the wrapped filters results
        :return:
        """

        # Let's inflate the clusters dictionary to link bounding boxes inside.
        clusters = {cluster_age_range: {'weight': weight, 'age_ranges': []}
                    for cluster_age_range, weight in self.clusters.items()}

        for (result, weight, _, age_range) in self.filter_result_list:

            if not result:
                continue

            # Are there any cluster in the list to match the age_range?
            clustered = False

            for cluster_age_range in clusters:

                intersection_rate = self._get_intersect_rate(cluster_age_range, age_range)

                if intersection_rate >= self.cluster_match_rate:
                    clusters[cluster_age_range]['weight'] += weight
                    clusters[cluster_age_range]['age_ranges'].append(age_range)
                    clustered = True
                    break

            if not clustered:
                clusters[age_range] = {'weight': weight, 'age_ranges': []}

        # Now we deflate the clusters by calculating the NMS of the bounding boxes linked together.
        self.clusters = {self._non_maximum_suppresion([age_range]+content['age_ranges']): content['weight']
                         for age_range, content in clusters.items()}

    @staticmethod
    def _non_maximum_suppresion(age_ranges):
        """
        Computes the non maximum suppression of a list of age ranges.
        :param age_ranges:
        :return:
        """
        minimum_x = min([age_range.get_range()[0] for age_range in age_ranges])
        maximum_y = max([age_range.get_range()[1] for age_range in age_ranges])

        return AgeRange(minimum_x, maximum_y)

    def _get_intersect_rate(self, age_range1, age_range2):
        """
        Computes the intersection rate for both age ranges.
        :param age_range1: age_range to intersect from
        :param age_range2: age_range to intersect with
        :return: percent of intersection with the smallest age range
        """

        min_area = min([age_range1.get_distance(), age_range2.get_distance(), 1])

        if min_area == 0:
            min_area = 1

        intersection = age_range1.intersect_with(age_range2)
        distance = intersection.get_distance()

        # Hack to allow clustering when intersection distance is 0 but it is a valid age.
        if distance == 0 and intersection.get_mean() > 0:
            distance = 0.5

        return distance / min_area