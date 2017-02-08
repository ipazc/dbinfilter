#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import shutil
from main.dataset.dataset import Dataset, ExtensionSet

__author__ = "Ivan de Paz Centeno"


class RawCrawledDataset(Dataset):
    """
    Works with a raw crawled .zip dataset.
    """

    def __init__(self, root_folder, metadata_file=None, description="Raw crawled dataset."):
        """
        Initializes a folder for this dataset.
        :param root_folder: folder for storing the dataset.
        :param metadata_file:
        :param description:
        """
        if not metadata_file:
            metadata_file = os.path.join(root_folder, "metadata.json")

        Dataset.__init__(self, root_folder, metadata_file, description)

    def import_from_zip(self, zip_filename):
        """
        Initializes the current dataset folder from a zip file.
        This method will load the dataset.

        :param zip_filename: zip to load the dataset from.
        """
        shutil.unpack_archive(zip_filename, self.root_folder, 'zip')
        self.load_dataset()

    def load_dataset(self):
        """
        Loads the dataset from the specified root folder.
        """

        # Let's load the routes list. This way we can reference them easily by the metadata_file content.
        self.file_extensions = ExtensionSet([".jpg", ".jpe", ".jpeg", ".png"])
        self._load_routes()
        self._load_metadata_file()

    def _load_metadata_file(self):
        """
        Loads the metadata content doing some preprocessing to its content.
        """
        Dataset._load_metadata_file(self)

        self.metadata_content = "".join(self.metadata_content)
        if self.metadata_content == "":
            self.metadata_content = "{}"

        self.metadata_content = json.loads(self.metadata_content)

        if 'data' in self.metadata_content:
            self.metadata_content = self.metadata_content['data']
