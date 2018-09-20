from copy import deepcopy

import rastervision as rv
from rastervision.data.label_source import (LabelSourceConfig,
                                            LabelSourceConfigBuilder,
                                            SemanticSegmentationRasterSource)
from rastervision.protos.label_source_pb2 import LabelSourceConfig as LabelSourceConfigMsg


class SemanticSegmentationRasterSourceConfig(LabelSourceConfig):
    def __init__(self, source, source_class_map):
        super().__init__(source_type=rv.SEMANTIC_SEGMENTATION_RASTER_SOURCE)
        self.source = source
        self.source_class_map = source_class_map

    def to_proto(self):
        msg = super().to_proto()
        opts = LabelSourceConfigMsg.SemanticSegmentationRasterSource(
            source=self.source.to_proto(),
            class_items=self.source_class_map.to_proto())
        msg.semantic_segmentation_raster_source.CopyFrom(opts)
        return msg

    def create_source(self, task_config, extent, crs_transformer, tmp_dir):
        return SemanticSegmentationRasterSource(self.source,
                                                self.source_class_map)

    def preprocess_command(self, command_type, experiment_config, context=[]):
        if context is None:
            context = []
        context = context + [self]
        io_def = rv.core.CommandIODefinition()

        b = self.to_builder()

        (new_raster_source,
         sub_io_def) = self.raster_source.preprocess_command(
             command_type, experiment_config, context)
        io_def.merge(sub_io_def)
        b = b.with_raster_source(new_raster_source)

        return (b.build(), io_def)


class SemanticSegmentationRasterSourceConfigBuilder(LabelSourceConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'source': prev.source,
                'source_class_map': prev.source_class_map
            }

        super().__init__(SemanticSegmentationRasterSourceConfig, config)

    def from_proto(self, msg):
        b = SemanticSegmentationRasterSourceConfigBuilder()

        return b \
            .with_raster_source(msg.semantic_segmentation_raster_source.source) \
            .with_source_class_map(
                msg.semantic_segmentation_raster_source.source_class_map)

    def with_raster_source(self, source):
        b = deepcopy(self)
        b.config['source'] = source
        return b

    def with_source_class_map(self, source_class_map):
        b = deepcopy(self)
        b.config['source_class_map'] = source_class_map
        return b
