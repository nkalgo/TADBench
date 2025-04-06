import tensorflow as tf

__all__ = [
    'default_kernel_initializer',
]


def default_kernel_initializer(weight_norm=False):
    """
    Get the default initializer for layer kernels (i.e., `W` of layers).

    Args:
        weight_norm: Whether or not to apply weight normalization
            (Salimans & Kingma, 2016) on the kernel?  If is not :obj:`False`
            or :obj:`None`, will use ``tf.random_normal_initializer(0, .05)``.

    Returns:
        The default initializer for kernels.
    """
    if weight_norm not in (False, None):
        return tf.random_normal_initializer(0., .05)
    else:
        '''
        它从以0为中心的截断的正态分布中提取样本，
        标准偏差(截断后)由stddev = sqrt(2 / (fan_in + fan_out))给出，
        其中fan_in是权张量中的输入单元数，fan_out是权张量中的输出单元数。
        '''
        return tf.glorot_normal_initializer()
