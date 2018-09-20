import unittest

import numpy as np

from rastervision.core.box import Box
from rastervision.core.class_map import ClassMap, ClassItem
from rastervision.data.label_source.semantic_segmentation_raster_source import (
    SemanticSegmentationRasterSource)
from rastervision.data.raster_source.raster_source import RasterSource


class MockRasterSource(RasterSource):
    def __init__(self, zeros=False, data=None):
        if data is not None:
            self.data = data
            (self.height, self.width, self.channels) = data.shape
        elif zeros and data is None:
            self.width = 4
            self.height = 4
            self.channels = 3
            self.data = np.zeros(
                (self.height, self.width, self.channels), dtype=np.uint8)
        elif not zeros and data is None:
            self.width = 4
            self.height = 4
            self.channels = 3
            self.data = np.random.randint(
                0,
                2,
                size=(self.width, self.height, self.channels),
                dtype=np.uint8)
            self.data[:, :, 0:(self.channels - 1)] = np.zeros(
                (self.height, self.width, self.channels - 1), dtype=np.uint8)

    def get_extent(self):
        return Box(0, 0, self.height, self.width)

    def _get_chip(self, window):
        ymin = window.ymin
        xmin = window.xmin
        ymax = window.ymax
        xmax = window.xmax
        return self.data[ymin:ymax, xmin:xmax, :]

    def get_chip(self, window):
        return self.get_chip(window)

    def get_crs_transformer(self, window):
        return None

    def get_dtype(self):
        return np.uint8


class TestSemanticSegmentationRasterSource(unittest.TestCase):
    def test_window_predicate_true(self):
        data = np.zeros((10, 10, 3), dtype=np.uint8)
        data[4:, 4:, :] = [1, 1, 1]
        raster_source = MockRasterSource(data=data)
        source_class_map = ClassMap([ClassItem(id=1, color='#010101')])
        label_store = SemanticSegmentationRasterSource(
            source=raster_source, source_class_map=source_class_map)
        extent = Box(0, 0, 10, 10)
        self.assertTrue(label_store.enough_target_pixels(extent, 30, [1]))

    def test_enough_target_pixels_false(self):
        data = np.zeros((10, 10, 3), dtype=np.uint8)
        data[7:, 7:, :] = [1, 1, 1]
        raster_source = MockRasterSource(data=data)
        source_class_map = ClassMap([ClassItem(id=1, color='#010101')])
        label_store = SemanticSegmentationRasterSource(
            source=raster_source, source_class_map=source_class_map)
        extent = Box(0, 0, 10, 10)
        self.assertFalse(label_store.enough_target_pixels(extent, 30, [1]))


if __name__ == '__main__':
    unittest.main()
