# Copyright 2019 Graphcore Ltd.
import inspect
import unittest
import os
import sys

from tests.resource_util import fetch_resources, captured_output
from classify_images import ImageClassifier


def parse_results_for_matching_tag(output, image_tag):
    """This function is extremely reliant on the output format of
        resnet_18/classify_images.py"""
    match = False
    if image_tag in output:
        top_match = output.split("\n")[1]
        if image_tag in top_match:
            match = True
    return match


class TestResnet18(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        fetch_resources('get_images_and_weights.sh',
                        os.path.join(cls.cwd, 'images', 'zebra.jpg'),
                        cls.cwd)
        cls.classify_img = ImageClassifier(os.path.join(cls.cwd, 'weights'))


    def verify_image(self, image_name):
        img = os.path.join(self.cwd, 'images', image_name + ".jpg")
        with captured_output() as out:
            self.classify_img.classify_image(img)

        output = out.getvalue().strip()
        self.assertTrue(parse_results_for_matching_tag(output, image_name))


    def test_inference_zebra(self):
        self.verify_image("zebra")


    def test_inference_pelican(self):
        self.verify_image("pelican")


    def test_inference_castle(self):
        self.verify_image("castle")


if __name__ == '__main__':
    unittest.main()
