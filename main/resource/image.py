#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import os
import cv2
import numpy
from main.resource.resource import Resource

__author__ = 'Iván de Paz Centeno'


class Image(Resource):
    """
    Represents an image. It is capable of storing the content in memory and perform some basic operations and checks
    on it.
    """

    def __init__(self, uri="", image_id="", metadata=None, blob_content=None):
        """
        Initialization of the image. The parameters of the image are read only, like on resources.
        This forces the creation of a new Image when a parameter changes. This allows to track or compare
        with previous states of the image. The exception is the blob_content.

        :param uri: URI to the image. It is a READ-ONLY property.
        :param image_id: ID of the image. It is a READ-ONLY property.
        :param metadata: metadata of the image. It is a READ-ONLY property.
        :param blob_content: content of the image (usually numpy array). This property is modificable.
        """
        Resource.__init__(self, uri=uri, res_id=image_id, metadata=metadata)

        self.cached_is_boolean_image = False
        self.cached_image_hash = None
        self.blob_content = None

        if blob_content is None:
            blob_content = []

        self.update_blob(blob_content)

    def save_to_uri(self):
        """
        Saves the current blob to the image URI. If the folder destination does not exist,
        this method will create it.
        """
        path, filename = os.path.split(self.get_uri())

        if not os.path.exists(path):
            os.mkdir(path)

        cv2.imwrite(self.uri, self.blob_content)

    def crop_image(self, bounding_box, new_uri):
        """
        Crops the current image and generates a new one with the cropped section.

        :param bounding_box: bounding box to crop
        :param new_uri: new uri to set to the resulting image.
        :return: A new image object with the cropped content. The bounding box is associated as the metadata of the
        image. Also, the image_id will be "cropped"
        """
        numpy_format = bounding_box.get_numpy_format()

        cropped_image = self.blob_content[numpy_format[0]:numpy_format[1],
                                          numpy_format[2]:numpy_format[3]]

        # FIX for C-Contiguous problem with the calculation of the hash md5.
        cropped_image = numpy.ascontiguousarray(cropped_image, cropped_image.dtype)

        return Image(uri=new_uri, image_id="cropped", metadata=[bounding_box], blob_content=cropped_image)

    def load_from_uri(self, as_gray=False):
        """
        Loads the blob from the URI.
        If the image couldn't be loaded, then is_load() method will return False.
        """
        uri = self.uri
        color_flag = {False: cv2.IMREAD_COLOR, True: cv2.IMREAD_GRAYSCALE}[as_gray]

        try:
            blob_content = cv2.imread(uri, color_flag)

        except Exception as ex:
            uri = os.fsencode(uri).decode('utf-8')
            blob_content = cv2.imread(uri, color_flag)

        if blob_content is None:
           blob_content = []

        self.update_blob(blob_content)

    def is_gray(self):
        """
        Checks whether the current image is in gray scale or not.

        :return: True if it's in gray scale, False otherwise.
        """
        is_gray = False

        if self.is_loaded():
            is_gray = len(self.blob_content.shape) == 2

        return is_gray

    def is_loaded(self):
        """
        Checks if the image is loaded or not.
        :return: True if is loaded into memory, False otherwise.
        """
        return self.blob_content is not None and len(self.blob_content) > 0

    def get_blob(self, as_rgb=False):
        """
        Getter for the blob of the image.
        :param as_rgb: sometimes the image is loaded in grayscale and it is required in RGB format. If this flag is
                       set, a channel is added to the image when it is in grayscale.
        :return: the blob content.
        """
        if as_rgb and self.is_gray():
            blob = cv2.cvtColor(self.blob_content, cv2.COLOR_GRAY2RGB)

        else:
            blob = self.blob_content

        return blob

    def get_size(self):
        """
        :return: size of the image in [width, height] format.
        """
        size = ()
        if self.is_loaded():
            size = self.blob_content.shape[0:2][::-1]  # We reverse the order row, cols in order to get x, y

        return size

    def update_blob(self, new_blob):
        """
        Updates the blob of the image.
        *Warning!* this method resets the flag that boolean saves that the image's pixels are in boolean format.
        If the blob is formed by boolean pixels, you must call convert_to_boolean() method again!.
        :param new_blob: updated blob of the image.
        """
        self.blob_content = new_blob

        if new_blob is not None and len(new_blob) > 0:
            self.cached_image_hash = hashlib.md5(self.blob_content).hexdigest()

        else:
            id = "{}, {}, {}".format(self.uri, self.res_id, self.metadata).encode("UTF-8")
            self.cached_image_hash = hashlib.md5(id).hexdigest()

    def convert_to_boolean(self):
        """
        Converts the image pixels into an array of boolean pixels.
        """
        if self.is_loaded():
            self.update_blob(self.blob_content > 0)

    def is_boolean(self):
        """
        Getter for the boolean flag.
        :return: True if image is set of boolean pixels. False if it is not.
        """
        return self.blob_content.dtype == 'bool'

    def convert_to_uint(self):
        """
        Converts the image into unsigned integers with 8 bits.
        """
        if self.is_loaded():
            self.update_blob(numpy.asarray(self.blob_content, dtype=numpy.uint8))

    def __str__(self):
        """
        :return: a string representation of the content.
        """
        return "Image, Loaded: {}, size: {}".format(self.is_loaded(), self.get_size())

    def md5hash(self):
        """
        :return: the md5hash for the image content.
        """
        return self.cached_image_hash

    def get_jpeg(self):
        """
        :return: returns the image binary content in jpeg format.
        """

        encoded_image = 0

        if self.is_loaded():
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            result, encimg = cv2.imencode('.jpg', self.blob_content, encode_param)

            if result:
                encoded_image = encimg.tostring()

        return encoded_image
