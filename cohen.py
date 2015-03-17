# -*- mode: Python; coding: utf-8 -*-
__author__ = 'Zachary Yocum'
__email__ = 'zyocum@brandeis.edu'

"""Functions for computing Cohen's kappa (𝜅), a measure of inter-annotator 
agreement between exactly two annotators."""

import numpy as np

def kappa(data):
    """Computes Cohen's 𝜅 coefficient given a confusion matrix.
    
        Pr(a) - Pr(e)
    𝜅 = -------------
          1 - Pr(e)
    
    Where Pr(a) is the percentage of observed agreement and Pr(e) is percentage 
    of expected agreement."""
    if len(data.shape) != 2:
        raise ValueError, 'input must be 2-dimensional array'
    if len(set(data.shape)) > 1:
        message = 'array dimensions must be N x N (they are {} x {})'
        raise ValueError, message.format(*data.shape)
    if not issubclass(data.dtype.type, np.integer):
        raise TypeError, 'expected integer type'
    if not np.isfinite(data).all():
        raise ValueError, 'all data must be finite'
    if (data < 0).any():
        raise ValueError, 'all data must be non-negative'
    if np.sum(data) <= 0:
        raise ValueError, 'total data must sum to positive value'
    observation = observed(data)
    expectation = expected(data)
    perfection = 1.0
    k = np.divide(
        observation - expectation,
        perfection - expectation
    )
    return k

def observed(data):
    """Computes the observed agreement, Pr(a), between annotators."""
    total = float(np.sum(data))
    agreed = np.sum(data.diagonal())
    percent_agreement = agreed / total
    return percent_agreement

def expected(data):
    """Computes the expected agreement, Pr(e), between annotators."""
    total = float(np.sum(data))
    annotators = range(len(data.shape))
    percentages = ((data.sum(axis=i) / total) for i in annotators) 
    percent_expected = np.dot(*percentages)
    return percent_expected

if __name__ == '__main__':
    # Example data from Pustejovsky and Stubbs
    # Natural Language Annotation for Machine Learning, (p. 127-128)
    #    B   B   B
    #    +   /   -
    data = np.array([
        [54, 28, 3 ], # + A
        [31, 18, 23], # / A
        [0,  21, 72]  # - A
    ])
    labels = ('Matrix', 'Pr(a)', 'Pr(e)', 'kappa')
    separator = ' : '
    width = max(map(len, labels))
    indent = ' ' * (width + len(separator))
    matrix = ('\n' + indent).join(map(str, data))
    values = (matrix, observed(data), expected(data), kappa(data))
    output = ''.join(('{:>', str(width), '}', separator, '{}'))
    print '\n'.join(output.format(*record) for record in zip(labels, values))