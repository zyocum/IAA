#!/usr/bin/env python3
# -*- mode: Python; coding: utf-8 -*-
"""Functions for computing Cohen's kappa (𝜅), a measure of inter-annotator 
agreement between exactly two annotators."""

import numpy as np

def kappa(data):
    """Computes Cohen's 𝜅 coefficient given a confusion matrix.
    
    The data must be an N x N matrix where N is the number of labels.
    
                 Pr(a) - Pr(e)
    Cohen's 𝜅 = ---------------
                   1 - Pr(e)
    
    Where...
        Pr(a) = the percentage of observed agreement
        Pr(e) = the percentage of expected agreement."""
    if not issubclass(data.dtype.type, np.integer):
        raise TypeError('expected integer type')
    if len(data.shape) != 2:
        raise ValueError('input must be 2-dimensional array')
    if len(set(data.shape)) > 1:
        message = 'array dimensions must be N x N (they are {} x {})'
        raise ValueError(message.format(*data.shape))
    if not np.isfinite(data).all():
        raise ValueError('all data must be finite')
    if (data < 0).any():
        raise ValueError('all data must be non-negative')
    if np.sum(data) <= 0:
        raise ValueError('total data must sum to positive value')
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