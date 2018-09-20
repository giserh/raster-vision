import tempfile

import numpy as np
import rasterio

from rastervision.utils.files import (get_local_path, make_dir, upload_or_copy)
from rastervision.data.label import SemanticSegmentationLabels
from rastervision.data.label_store import LabelStore
from rastervision.data.utils import SegmentationClassTransformer


class SemanticSegmentationRasterStore(LabelStore):
    """A prediction label store for segmentation raster files.
    """

    def __init__(self, uri, class_map):
        self.uri = uri
        self.class_transfomer = SegmentationClassTransformer(class_map)

    def get_labels(self):
        """Get all labels.

        Returns:
             np.ndarray
        """
        # TODO: build it
        source = None
        rgb_labels = source.get_raw_image_array()
        return self.class_transformer.transform(rgb_labels)

    def save(self, labels):
        """Save.

        Args:
           labels - Labels to be saved, the type of which will be dependant on the type
                    of task.
        """
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name
        local_path = get_local_path(self.uri, temp_dir)
        make_dir(local_path, use_dirname=True)

        extent = labels.get_extent()

        # https://github.com/mapbox/rasterio/blob/master/docs/quickstart.rst
        # https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html
        with rasterio.open(
                local_path,
                'w',
                driver='GTiff',
                height=extent.ymax,
                width=extent.xmax,
                count=3,
                dtype=np.uint8) as dataset:
            for (window, class_labels) in self.label_pairs:
                window = (window.ymin, window.ymax), (window.xmin, window.xmax)
                rgb_labels = self.class_transformer.class_to_rgb(class_labels)
                for chan in range(3):
                    dataset.write_band(
                        chan + 1, rgb_labels[chan, :, :], window=window)

        upload_or_copy(local_path, self.uri)

    def empty_labels(self):
        """Produces an empty Labels"""
        return SemanticSegmentationLabels()
