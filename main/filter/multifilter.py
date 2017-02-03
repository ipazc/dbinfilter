#!/usr/bin/env python
# -*- coding: utf-8 -*-

from main.filter.filter import Filter
from multiprocessing import Pool

__author__ = "Ivan de Paz Centeno"


def apply_filter(resource_mapped_filter):
    """
    Multiprocessed function to apply a given filter to a specified resource.
    :param resource_mapped_filter:
    :return:
    """
    _filter = resource_mapped_filter[0]
    resource = resource_mapped_filter[1]

    return _filter.apply_to(resource)


class Multifilter(Filter):
    """
    Wraps a set of filters in order to apply them all together in parallel.
    It is way faster than a list comprehension or a common loop.
    """

    def __init__(self, filter_list):
        super().__init__(0)

        if filter_list is None:
            filter_list = []

        self.filter_list = filter_list
        self.pool = Pool(len(self.filter_list))

    def apply_to(self, resource):
        """
        Applies the set of filters to the given resource.
        :param resource:
        :return:
        """
        resource_mapped_filter_list = [[_filter, resource] for _filter in self.filter_list]
        return self.pool.map(apply_filter, resource_mapped_filter_list)

    def apply_to_list(self, resource_list):
        """
        Applies the set of filters to the specified set of resources.
        :param resource:
        :return:
        """
        resource_mapped_filter_list = [[_filter, resource] for _filter in self.filter_list for resource in resource_list]
        return self.pool.map(apply_filter, resource_mapped_filter_list)
