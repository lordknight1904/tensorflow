import tensorflow as tf


class Dataset(tf.data.Dataset):
    inter = None

    def __init__(
            self,
            batch_size=tf.placeholder(tf.int64),

        ):
        tf.data.Dataset.__init__(self)
        pass

    def _as_variant_tensor(self):
        pass

    def _inputs(self):
        pass

    def output_classes(self):
        pass

    def output_shapes(self):
        pass

    def output_types(self):
        pass
