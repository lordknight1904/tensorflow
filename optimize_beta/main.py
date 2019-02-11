import configparser as cp
import tensorflow as tf
from optimize_beta.core.extractor import Extractor


if __name__ == '__main__':
    # cfg = cp.ConfigParser()
    # cfg.read('./core/config.ini')
    # config = cfg['DEFAULT']
    # print(config.get('ServerAliveInterval'))

    extractor = Extractor(
        [{'name': 'SP500', 'type': 'csv'}],
    )
    iterator = extractor.dataset.take(10).make_one_shot_iterator()
    next_element = iterator.get_next()
    sess = tf.Session()

    while True:
        try:
            x = sess.run(next_element)
            print(x)
        except tf.errors.OutOfRangeError:
            break

