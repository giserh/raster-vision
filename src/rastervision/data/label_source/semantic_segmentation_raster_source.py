from typing import (List, Union)

import numpy as np

from rastervision.core.box import Box
from rastervision.data.utils import SegmentationClassTransformer
from rastervision.data.label_source import LabelSource
from rastervision.data.raster_source import RasterSource


class SemanticSegmentationRasterSource(LabelSource):
    """A read-only label source for segmentation raster files.
    """

    def __init__(self, source: RasterSource, source_class_map):
        """Constructor.

        Args:
             source: A source of raster label data (either an object that
                  can provide it or a path).
        """
        # TODO handle different ways of specifying source
        self.source = source
        self.class_transformer = SegmentationClassTransformer(source_class_map)

    def enough_target_pixels(self, window: Box, target_count_threshold: int,
                             target_classes: List[int]) -> bool:
        """Given a window, answer whether the window contains enough pixels in
        the target classes.

        Args:
             window: The larger window from-which the sub-window will
                  be clipped.
             target_count_threshold:  Minimum number of target pixels.
             target_classes: The classes of interest.  The given
                  window is examined to make sure that it contains a
                  sufficient number of target pixels.
        Returns:
             True (the window does contain interesting pixels) or False.
        """
        rgb_labels = self.source.get_raw_chip(window)
        labels = self.class_transformer.rgb_to_class(rgb_labels)

        target_count = 0
        for class_id in target_classes:
            target_count = target_count + (labels == class_id).sum()

        return target_count >= target_count_threshold

    def get_labels(self, window: Union[Box, None] = None) -> np.ndarray:
        """Get labels from a window.

        If self.source is not None then a label window is clipped from
        it.  If self.source is None then assume window is full extent.

        Args:
             window: Either None or a window given as a Box object.
        Returns:
             np.ndarray
        """
        if window is None:
            rgb_labels = self.source.get_raw_image_array()
        else:
            rgb_labels = self.source.get_raw_chip(window)

        return self.class_transformer.rgb_to_class(rgb_labels)
