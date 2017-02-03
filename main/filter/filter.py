#!/usr/bin/env python
# -*- coding: utf-8 -*-
from main.resource.resource import Resource

__author__ = "Ivan de Paz Centeno"


class Filter(object):
    """
    Allows to apply a filter to a resource.
    """

    def __init__(self, weight):
        """
        Initializes the filter with a weight.
        :param weight: confidence weight for this filter. Useful for inferences.
        """
        self.weight = weight

    def apply_to(self, resource):
        """
        Applies the filter to the resource
        :param resource: resource to filter.
        :return: a filter score.
        """
        pass

    def get_weight(self):
        """
        getter for the weight
        :return:
        """
        return self.weight

    def get_type(self):
        """
        Shows the prototype of the type of resource that this filter accepts.
        :return:
        """
        return Resource


FILTERS_PROTO={}