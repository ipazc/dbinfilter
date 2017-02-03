#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Iván de Paz Centeno'

import fnmatch
import os
import errno


LMDB_BATCH_SIZE = 256    # Batch size for writing into LMDB. This is the amount of image
                         # before the batch is commited into the file.


def mkdir_p(dir):
    """
    Creates a dir recursively (like mkdir -p).
    If it already exists does nothing.
    :param dir: dir to create.
    :return:
    """

    try:
        os.makedirs(dir)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(dir):
            pass
        else:
            print("Error when creating dir for dataset: {}".format(exc))
            raise


class Dataset(object):
    """
    Abstract class representing a dataset.
    Provides basic dataset functionality, like iterating over a folder in order to retrieve files or read
    a metadata file into memory.
    """

    def __init__(self, root_folder, metadata_file, description):
        """
        Initializes the dataset object.
        Override this constructor to append functionality, for example loading the folders upon initialization with
        _load_routes() method.
        :param root_folder: root folder of the dataset.
        :param metadata_file: file that represents the metadata (usually labels or ground truth)
        :param description: basic description of the dataset. This is useful for reports.
        :return: instance of the class Dataset.
        """
        self.root_folder = root_folder
        self.metadata_file = metadata_file
        self.description = description
        self.file_extensions = ExtensionSet([".jpg"])
        self.metadata_content = {}
        self.routes = []

    def _load_routes(self):
        """
        Iterates over the folder recursively in order to retrieve the files that matches the file extensions
        """
        for root, dirnames, filenames in os.walk(self.root_folder):
            for filename in self.file_extensions.fnfilter(filenames):
                self.routes.append(os.path.join(root, filename))

    def _load_metadata_file(self):
        """
        Reads the metadata file into the memory.
        It can be accessed through self.metadata_content
        """
        if not os.path.exists(self.metadata_file):
            self.metadata_content = [""]
            return

        with open(self.metadata_file) as file:
            self.metadata_content = file.readlines()

    def get_routes(self):
        """
        Getter for the routes.
        :return:
        """
        return list(self.routes)

    def find_route(self, token_id):
        """
        finds a route given a specific ID, usually retrieved from the metadata.
        :param token_id: id token to search for.
        :return: routes that matches the specified id token. It may return more than one route if
        the ID is not specific enough.
        """
        search_key = token_id
        return [route for route in self.routes if search_key in route]

    def get_root_folder(self):
        """
        Getter for the root folder.
        """
        return self.root_folder

    def get_metadata_filename(self):
        """
        Getter for the metadata filename.
        """
        return self.metadata_file

    def get_description(self):
        """
        Getter for the dataset description.
        """
        return self.description

    # each class must override this method since the key to search
    # could be different in each case.
    # def find_image_route(self, image_id):

    def get_metadata_content(self):
        """
        Getter for the metadata content.
        It is encouraged to have called the _load_metadata_file() at least once before calling this method.
        """
        return self.metadata_content


def preprocess_extension(func):
    """
    Decorator for extension string.

    Removes the initial "." if it exists.

    """

    def func_wrapper(extension):
        """
        :param extension: extension to preprocess.
        """

        return extension.replace(".", "")

    return func_wrapper


class ExtensionSet(object):
    """
    Provides a set of extensions and a formatter to become a search filter.
    """

    def __init__(self, extension_list=None):
        """
        Constructor of the extension set class.
        :param extension_list: list of extensions to initialize the object.
        """

        if extension_list is None:
            extension_list = []

        extension_list = [extension.replace(".", "") for extension in extension_list]
        self.extensions = set(extension_list)  # Clone value

    def to_list(self):
        """
        Returns a list of extensions stored in the object.
        :return: list of extensions (cloned from original).
        """
        return list(self.extensions)

    @preprocess_extension
    def add_extension(self, extension):
        """
        Appends a new extension to the set.
        :param extension: extension to append, with/without the initial "."
        :return: True if extension was added, False if extension already exists in the set.
        """
        already_exists = extension in self.extensions
        self.extensions.add(extension)

        return not already_exists

    @preprocess_extension
    def remove_extension(self, extension):
        """
        Removes the extension from the list.
        :param extension: extension to remove from the set.
        :return: True if extension could be removed. False otherwise.
        """
        already_exists = extension in self.extensions
        self.extensions.remove(extension)

        return already_exists

    def fnfilter(self, filenames):
        """
        filters the filenames list to retrieve the ones that matches these extensions.
        :return list of files that matches these extensions.
        """
        return list(filter(None, [",".join(fnmatch.filter(filenames,'*.{}'.format(extension)))
                                  for extension in [self.extensions]]))


dataset_proto = {}
