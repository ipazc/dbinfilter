#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Ivan de Paz Centeno"


class FilterCompositionClustering(object):
    """
    Clusterizes the result of a set of filters
    """

    def __init__(self, filter_result_list, cluster_match_rate=0.5):
        """
        List of results of the filters.
        Example: [(filter_result, filter_weight, filter_reason, filter_bounding_boxes)]
        :param filter_result_list:
        """
        self.filter_result_list = filter_result_list
        self.cluster_match_rate = cluster_match_rate
        self.clusters = {}

    def find_clusters(self):
        pass

    def get_found_clusters(self):
        return self.clusters