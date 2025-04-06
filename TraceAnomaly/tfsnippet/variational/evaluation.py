import tensorflow as tf

from tfsnippet.ops import log_mean_exp
from .utils import _require_multi_samples

__all__ = ['importance_sampling_log_likelihood',
           'importance_sampling_log_likelihood_per_dimension']


def importance_sampling_log_likelihood(log_joint, latent_log_prob, axis,
                                       keepdims=False, name=None):
    """
    Compute :math:`\\log p(\\mathbf{x})` by importance sampling.

    .. math::

        \\log p(\\mathbf{x}) =
            \\log \\mathbb{E}_{q(\\mathbf{z}|\\mathbf{x})} \\Big[\\exp\\big(\\log p(\\mathbf{x},\\mathbf{z}) - \\log q(\\mathbf{z}|\\mathbf{x})\\big) \\Big]

    Args:
        log_joint: Values of :math:`\\log p(\\mathbf{z},\\mathbf{x})`,
            computed with :math:`\\mathbf{z} \\sim q(\\mathbf{z}|\\mathbf{x})`.
        latent_log_prob: :math:`q(\\mathbf{z}|\\mathbf{x})`.
        axis: The sampling dimensions to be averaged out.
        keepdims (bool): When `axis` is specified, whether or not to keep
            the averaged dimensions?  (default :obj:`False`)
        name (str): TensorFlow name scope of the graph nodes.
            (default "importance_sampling_log_likelihood")

    Returns:
        The computed :math:`\\log p(x)`.
    """
    _require_multi_samples(axis, 'importance sampling log-likelihood')
    log_joint = tf.convert_to_tensor(log_joint)
    latent_log_prob = tf.convert_to_tensor(latent_log_prob)
    with tf.name_scope(name, default_name='importance_sampling_log_likelihood',
                       values=[log_joint, latent_log_prob]):
        log_p = log_mean_exp(
            log_joint - latent_log_prob, axis=axis, keepdims=keepdims)
        return log_p


def importance_sampling_log_likelihood_per_dimension(log_joint, latent_log_prob, axis, name=None):
    """
    Compute :math:`\\log p(\\mathbf{x})` per dimension by importance sampling.

    Args:
        log_joint: Values of :math:`\\log p(\\mathbf{z},\\mathbf{x})`,
            computed with :math:`\\mathbf{z} \\sim q(\\mathbf{z}|\\mathbf{x})`.
        latent_log_prob: :math:`q(\\mathbf{z}|\\mathbf{x})`.
        name (str): TensorFlow name scope of the graph nodes.
            (default "importance_sampling_log_likelihood_per_dimension")

    Returns:
        The computed :math:`\\log p(x)` per dimension.
    """
    log_joint = tf.convert_to_tensor(log_joint)
    latent_log_prob = tf.convert_to_tensor(latent_log_prob)
    with tf.name_scope(name, default_name='importance_sampling_log_likelihood_per_dimension',
                       values=[log_joint, latent_log_prob]):
        log_p = log_mean_exp(
            log_joint - latent_log_prob, axis=axis, keepdims=True)
        return log_p
