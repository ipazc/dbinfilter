#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os


__author__ = 'Ivan de Paz Centeno'


class Resource(object):
    """
    Represents a resource in the system (a file). This is a virtual class, do not use it directly.
    It may also contain the content of the file, or even represent a virtual file if the URI is not real.
    """
    def __init__(self, uri="", res_id="", metadata=None):
        """
        Instantiates a resource. This class shouldn't be instantiated directly, instead instantiate a class
        that inherits this one. If a parameter should be changed, then you need a new instantiation of the resource.

        :param uri: URI to the resource. READ-ONLY parameter after instantiation.
        :param res_id: ID of the resource. READ-ONLY parameter after instantiation.
        :param metadata: metadata asociated with the resource. READ-ONLY parameter after instantiation.
        """
        if metadata is None:
            metadata = []

        self.uri = uri
        self.res_id = res_id
        self.metadata = metadata

    def get_uri(self):
        """
        Getter for the URI
        :return: the URI of the resource
        """
        return self.uri

    def exists(self):
        """
        Checks if the file is virtual or not (URI exists)
        :return: True if file exists at the specified URI. False otherwise.
        """
        try:
            result = os.path.isfile(self.uri)
        except:
            uri = self.uri.encode("utf-8")
            result = os.path.isfile(uri)

        return result

    def get_id(self):
        """
        Getter for the ID of the resource.
        :return: ID of the resource.
        """
        return self.res_id

    def get_metadata(self):
        """
        Getter for the metadata of the resource (that is, the extra data associated with it).
        :return: Dictionary with extra data associated with this resource.
        """
        return self.metadata

    def __str__(self):
        """
        :return: String representation of the resource.
        """
        return "[{}: {}] \"{}\"; {} metadata elements".format(self, self.res_id, self.uri, len(self.metadata))