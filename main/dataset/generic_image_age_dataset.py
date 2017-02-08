#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import random
import caffe
from caffe.io import array_to_datum
from caffe.proto import caffe_pb2
import cv2
import lmdb
import numpy as np
from main.dataset.dataset import Dataset, mkdir_p, LMDB_BATCH_SIZE, dataset_proto
from main.resource.image import Image
from main.tools.age_range import AgeRange
import shutil

__author__ = 'Iván de Paz Centeno'


class GenericImageAgeDataset(Dataset):
    """
    Dataset of image for ages.
    It allows to read an existing dataset or to create a new one under the specified root folder.
    """

    def __init__(self, root_folder, metadata_file="labels.json",
                 description="Generic Dataset JSON-Based of image with Age labels", dataset_normalizers=None):
        """
        Initialization of a dataset of image with ages labeled.
        Metadata is built from a JSON file.
        :param root_folder:
        :param metadata_file: JSON file URI, which is composed by the following format: {'IMAGE_FILE_NAME': AGE_RANGE_STRING}

                              Example:
                                {'test/image1.jpg': '[23,34]'}

                              Note: the working directory when loading the metadata_file is root_folder.
        :param description: description of the dataset for report purposes.
        :param dataset_normalizers: list of normalizers to normalize when storing image inside this dataset.
        :return:
        """
        # This dataset class is also capable of creating datasets.
        mkdir_p(root_folder)

        # When metadata_file is not an absolute URI, it is related to root_folder.
        if not self._is_absolute_uri(metadata_file):
            metadata_file = os.path.join(root_folder, metadata_file)

        Dataset.__init__(self, root_folder, metadata_file, description)

        self.autoencoded_uris = {}

        if not dataset_normalizers:
            dataset_normalizers = []

        self.normalizers = dataset_normalizers

    def update_normalizers(self, dataset_normalizers):
        """
        Updates the normalizers for images from this dataset.
        :param dataset_normalizers: list of normalizers.
        :return:
        """
        self.normalizers = dataset_normalizers

    def _is_absolute_uri(self, uri):
        """
        Checks if the specified uri is absolute or relative.
        :param uri: URI to check
        :return: True if it is absolute, False otherwise.
        """
        return uri.startswith("/")

    def get_keys(self, shuffle=False):
        """
        Returns the keys for the find_route method as a list.
        :param shuffle: shuffles the list before returning it.
        :return: list of keys for the find_route
        """
        result = list(self.metadata_content.keys())

        if shuffle:
            random.shuffle(result)
            
        return result

    def get_key_metadata(self, key):
        """
        Retrieves the metadata for the specified key
        :param key:
        :return:
        """

        return self.metadata_content[key]

    def get_image(self, key):
        """
        Retrieves the image representing the specified key ID.
        :return:
        """
        # For generic image age dataset, the key is the relative uri to the file.
        uri = self._get_key_absolute_uri(key)
        image = Image(image_id=key, uri=uri, metadata=[self.get_key_metadata(key)])

        return image

    def _get_key_absolute_uri(self, key):
        """
        Retrieves the absolute uri for a key.
        :param key: key to retrieve uri from.
        :return: absolute uri of the key.
        """
        return os.path.join(self.root_folder, key)

    def put_image(self, image, autoencode_uri=True, apply_normalizers=True):
        """
        Puts an image in the dataset.
        It must be filled with content, relative uri and metadata in order to be created the dataset.
        :param image: Image to save in the dataset.
        :param autoencode_uri: Boolean flag to set if the URI should be automatically filled by the dataset or not.
        :param dataset_normalizer: normalizer to apply to the image
        :param apply_normalizers: boolean flag to apply normalizers when the image is put into the dataset manually.
        :return:
        """

        if autoencode_uri:
            uri = self._encode_uri_for_image(image)

        else:
            uri = image.get_uri()

        if self._is_absolute_uri(uri):
            raise Exception("Uri for storing into dataset must be relative, not absolute")

        key = uri   # We index by the relative uri
        uri = os.path.join(self.root_folder, uri)
        mkdir_p(os.path.dirname(uri))

        self.metadata_content[key] = image.get_metadata()[0]

        try:
            if not image.is_loaded():
                image.load_from_uri()

            image_blob = image.get_blob()

            normalizers_applied = 0
            if apply_normalizers:
                for normalizer in self.normalizers:
                    image_blob = normalizer.apply(image_blob)
                    normalizers_applied += 1

            cv2.imwrite(uri, image_blob)
            print("Saved into {} ({} normalizers applied)".format(uri, normalizers_applied))

        except Exception as ex:
            print("Could not write image \"{}\" into dataset.".format(image.get_uri()))
            del self.metadata_content[key]
            raise

    def put_resource(self, resource, autoencode_uri=True, apply_normalizers=True):
        """
        Puts the resource into an image and then pipes it to the put_image.
        :param resource:
        :param autoencode_uri:
        :param apply_normalizers:
        :return:
        """
        image = Image(uri=resource.get_uri(), metadata=resource.get_metadata())
        self.put_image(image, autoencode_uri=autoencode_uri, apply_normalizers=apply_normalizers)

    def _update_encoded_uris_cache(self):
        """
        Updates the encoded uris cache based on the current metadata.
        :return:
        """
        self.autoencoded_uris = {}

        keys = self.get_keys()

        for key in keys:

            metadata = self.get_key_metadata(key)
            age_range_hash = metadata.hash()

            if age_range_hash not in self.autoencoded_uris:
                self.autoencoded_uris[age_range_hash] = 1
            else:
                self.autoencoded_uris[age_range_hash] += 1

    def _encode_uri_for_image(self, image):
        """
        Finds a new URI for the specified image inside the dataset.
        It is going to create an uri based on the age-range and the number of image that matches that age range.
        :param image: image to find an URI for.
        :return: URI for the image.
        """
        try:
            age_range = image.get_metadata()[0]
        except Exception as ex:
            print("Image to save does not contain label for age (AgeRange) in metadata.")
            raise

        if age_range.hash() not in self.autoencoded_uris:
            self.autoencoded_uris[age_range.hash()] = 0
            #mkdir_p(os.path.join(self.root_folder, "{}-{}".format(age_range.get_range()[0], age_range.get_range()[1])))

        uri = "{}-{}/{}.jpg".format(age_range.get_range()[0], age_range.get_range()[1],
                                 self.autoencoded_uris[age_range.hash()])
        self.autoencoded_uris[age_range.hash()] += 1

        return uri

    def load_dataset(self):
        """
        Loads the dataset from the specified root folder.
        """

        # Let's load the routes list. This way we can reference them easily by the metadata_file content.
        self._load_routes()
        self._load_metadata_file()

        if "".join(self.metadata_content) == "":
            self.metadata_content = "{}"

        self.metadata_content = self._preprocess_metadata(self.metadata_content)
        self._update_encoded_uris_cache()

    @staticmethod
    def _preprocess_metadata(raw_metadata):
        """
        Processes the metadata content in order to be framework-compliant (AgeRange classes for ages)
        :param raw_metadata: raw metadata content to be processed.
        :return: Metadata dict with {FileName:AgeRange} format.
        """
        metadata_content = json.loads("".join(raw_metadata))
        preprocessed_metadata = {}

        for key, value in metadata_content.items():
            preprocessed_metadata[key] = AgeRange.from_string(value)

        return preprocessed_metadata

    @staticmethod
    def _postprocess_metadata(preprocessed_metadata):
        """
        Processes the metadata to make a raw metadata again (in order to be dumped to a file for example).
        :param preprocessed_metadata:
        :return: raw metadata without objects, dumpable into string and/or file.
        """
        postprocessed_metadata = {}

        for key, value in preprocessed_metadata.items():
            postprocessed_metadata[key] = value.to_dict()["Age_range"]

        return postprocessed_metadata

    def save_dataset(self):
        """
        Dumps the metadata labels in JSON format inside the dataset's folder with name labels.json
        :return:
        """

        with open(self.metadata_file, 'w') as outfile:
            json.dump(self._postprocess_metadata(self.metadata_content), outfile, indent=4)

    def get_dataset_size(self):
        """
        Calculates the dataset size based on the image' sizes.
        :return: size in bytes of the whole dataset (excluding the metadata).
        """
        keys = self.get_keys()

        dataset_size = 0
        for key in keys:
            image = self.get_image(key)
            image.load_from_uri()
            dataset_size += image.get_blob().nbytes

        return dataset_size

    def export_to_lmdb(self, lmdb_foldername, ages_as_means=True, map_size=-1, splitters=None, apply_normalizers=False):
        """
        Exports the current dataset to LMDB format.
        If the LMDB already exists, it will append to its content.
        :param lmdb_foldername: filename LMDB.
        :param ages_as_means: save the age_range in mean format.
        :param map_size: size of map of the LMDB database. If set to -1, it will attempt to calculate a map_size that
        fits this dataset. Remember however, that it won't allow to expand the LMDB database with new data and you'll
        need to create a new one in case you want to.
        :param splitters: a set of splitters to split the dataset into multiple lmdbs. The lmdb divided by each splitter
        will be stored with the splitter's name prepended to the lmdb name. This is useful if you want to extract a
        chunk of the dataset as a test or validation lmdbs.
        :param apply_normalizers: boolean flag to apply normalizers when the image is put into the dataset manually.
        """
        iteration = 0
        txn_index = 0

        if splitters is None:
            splitters = []

        keys = self.get_keys(shuffle=True)
        count = len(keys)
        datum_id_format = "{}:0>{}{}_dbuild_{}".format("{", len(str(count)), "}", "{}")

        if map_size == -1:
            #map_size = self.get_dataset_size() + count * 30000
            map_size = 1e12

        print("Map size is {} MBytes".format(round(map_size/1000/1000, 2)))

        if splitters:
            environments = [lmdb.Environment(lmdb_foldername + "_" + splitter.get_name(), map_size=map_size) for splitter in splitters]
        else:
            environments = [lmdb.Environment(lmdb_foldername, map_size=map_size)]

        txns = [env.begin(write=True, buffers=True) for env in environments]
        txn = env = None

        put_txns = {}

        for txn in txns:
            put_txns[txn] = 0

        for key in keys:
            iteration += 1

            for split_id in range(len(splitters)):

                splitter = splitters[split_id]

                if splitter.decide(iteration):
                    txn = txns[split_id]
                    env = environments[split_id]
                    txn_index = split_id
                    break

            if not txn or not env:
                txn = txns[0]
                env = environments[0]
                txn_index = 0

            image = self.get_image(key)
            image.load_from_uri()
            age_range = image.get_metadata()[0]

            if ages_as_means:
                label = age_range.get_mean()
            else:
                label = age_range.get_range()[0]

            image_blob = image.get_blob()
            if apply_normalizers:
                for normalizer in self.normalizers:
                    image_blob = normalizer.apply(image_blob)

            #HxWxC to CxHxW in caffe
            image_blob = np.transpose(image_blob, (2,0,1))

            # Datum is the element map in LMDB. We associate image with label here.
            datum = array_to_datum(image_blob, label)

            # Now we encode the image id in ascii format inside the lmdb container that corresponds to this input.

            txn.put(datum_id_format.format(iteration, image.get_id()).encode("ascii"), datum.SerializeToString())
            put_txns[txn] += 1

            # write batch
            if put_txns[txn] % LMDB_BATCH_SIZE == 0:
                txn.commit()
                old_txn_count = put_txns[txn]
                del put_txns[txn]
                txn = env.begin(write=True)
                txns[txn_index] = txn
                put_txns[txn] = old_txn_count
                print("[{}%] Stored batch of {} image in LMDB".format(round(iteration/count * 100, 2),
                                                                      LMDB_BATCH_SIZE))

        # There could be a last batch on each txn without being commited.
        [txn.commit() or print("[{}%] Stored batch of {} image in LMDB".format(round(iteration/len(keys) * 100, 2),
                               count % LMDB_BATCH_SIZE))
         for txn, count in put_txns.items() if count % LMDB_BATCH_SIZE != 0]

        [env.close() for env in environments]

    def import_from_lmdb(self, lmdb_foldername):
        """
        Imports the dataset from LMDB format into the root_folder.
        :param lmdb_foldername: filename LMDB.
        """

        lmdb_env = lmdb.open(lmdb_foldername)
        lmdb_txn = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()

        datum = caffe_pb2.Datum()

        for key, value in lmdb_cursor:
            key = str(key, encoding="UTF-8").split("_dbuild_", 1)[-1]

            datum.ParseFromString(value)

            label = datum.label
            data = caffe.io.datum_to_array(datum)

            # CxHxW to HxWxC in cv2
            image_blob = np.asarray(np.transpose(data, (1, 2, 0)), order='C')

            image = Image(uri=key, image_id=key, metadata=[AgeRange(label, label)], blob_content=image_blob)
            self.put_image(image, autoencode_uri=False)

        lmdb_env.close()

    def export_to_zip(self, filename):
        """
        Exports the current dataset into ZIP format.
        :param filename: filename of the zip to store contents into.
        """
        shutil.make_archive(filename, 'zip', self.root_folder)

    def import_from_zip(self, filename):
        """
        Imports the specified dataset from ZIP format. It won't delete current dataset's contents nor the config file.
        :param filename: filename of the zip to store contents into.
        """
        self.load_dataset()
        shutil.unpack_archive(filename, self.root_folder, 'zip')
        previous_metadata_content = self.metadata_content
        self.routes = []
        self.load_dataset()

        if not self.metadata_content:
            print("Warning: imported ZIP does not seem to have a valid metadata content.")

        for k,v in previous_metadata_content.items():
            self.metadata_content[k] = v

    def get_metadata_proto(self):
        """
        Retrieves the metadata proto used by this class.
        :return:
        """
        return AgeRange

    def clean(self, remove_files=False):
        """
        Cleans the current dataset storage.
        :param remove_files: boolean flag. If set, it will also be reflected in the filesystem.
        :return:
        """
        keys = self.get_keys()

        if remove_files:

            for key in keys:
                absolute_uri = self._get_key_absolute_uri(key)
                os.remove(absolute_uri)
                os.remove(self.metadata_file)

        self.metadata_content = {}

dataset_proto[GenericImageAgeDataset.__name__] = GenericImageAgeDataset
