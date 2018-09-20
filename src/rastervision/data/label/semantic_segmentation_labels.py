from rastervision.data.label import Labels
from rastervision.core.box import Box


class SemanticSegmentationLabels(Labels):
    """A set of spatially referenced labels.
    """

    def __init__(self):
        self.label_pairs = []

    def __add__(self, other):
        """Add labels to these labels.

        Returns a concatenation of this and the other labels.
        """
        self.label_pairs.extend(other.label_pairs)

    def add_label_pair(self, window, label_array):
        self.label_pairs.append((window, label_array))

    def get_label_pairs(self):
        return self.label_pairs

    def get_extent(self):
        windows = list(map(lambda pair: pair[0], self.get_label_pairs()))
        xmax = max(map(lambda w: w.xmax, windows))
        ymax = max(map(lambda w: w.ymax, windows))
        return Box(0, 0, ymax, xmax)
