#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from main.resource.resource import Resource

__author__ = 'Ivan de Paz Centeno'


class Text(Resource):

    def __init__(self, uri="", text_id="", metadata=None, content=None):
        Resource.__init__(self, uri=uri, res_id=text_id, metadata=metadata)

        if content is None:
            content = []

        self.content = content

    def __str__(self):
        return "[Text {}] \"{}\"; {} metadata elements".format(self.res_id, self.uri, len(self.metadata))

    def load_from_uri(self):
        with open(self.uri, 'r', encoding='utf-8') as temp_file:
            self.content = temp_file.readlines()

    def is_loaded(self):
        return len(self.content) > 0

    def get_content(self):

        return self.content