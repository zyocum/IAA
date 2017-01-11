#!/usr/bin/env python3
# -*- mode: Python; coding: utf-8 -*-
"""Functions for computing Fleiss's kappa (𝜅), a measure of inter-annotator 
agreement between several (more than 2) annotators."""

import numpy as np

def kappa(data):
    """Computes Fleiss's 𝜅 coefficient given a matrix of subject ratings.
    
    The data must be an N x M matrix where N is the number of subjects and M is
    the number of labels.
    
                  P̅ - P̅(e)
    Fleiss's 𝜅 = ----------
                  1 - P̅(e)
    
    Where...
        P̅    = the percentage of observed agreement
        P̅(e) = the percentage of expected agreement."""
    if not issubclass(data.dtype.type, np.integer):
        raise TypeError('expected integer type')
    if len(data.shape) != 2:
        raise ValueError('input must be 2-dimensional array')
    if not np.isfinite(data).all():
        raise ValueError('all data must be finite')
    if (data < 0).any():
        raise ValueError('all data must be non-negative')
    if np.sum(data) <= 0:
        raise ValueError('total data must sum to positive value')
    if not len(set(sum(data.T))) == 1:
        raise ValueError('all subjects must have the same number of ratings')
    observation = observed(data)
    expectation = expected(data)
    perfection = 1.0
    k = np.divide(
        observation - expectation,
        perfection - expectation
    )
    return k

def label_proportions(data):
    """Computes the proportion of ratings, p(j), assigned to each label.
    
    I.e., for each label j, what percentage of all ratings, were
    assigned to that label."""
    label_proportions = np.divide(sum(data).astype(float), data.sum())
    return label_proportions

def subject_agreements(data):
    """Computes the per-subject agreement, P(i), for each subject.
    
    I.e., compute how many inter-rater pairs are in agreement, relative to the
    number of all possible pairs:
    
           𝛴(j=1 → k)[n(i,j)² - n]
    P(i) = -----------------------
                  n(n - 1)
    
    Where...
        i      = a subject (i.e., row) index
        j      = a label (i.e., column) index
        k      = the number of labels
        n(i,j) = how many raters assigned the j-th label to the i-th subject
        n      = number of ratings per subject (i.e., the sum of one row)"""
    subject_indices, label_indices = map(range, data.shape)
    dot_products = np.array([np.dot(data[i], data[i]) for i in subject_indices])
    sums = data.sum(axis=1).astype(float)
    numerators = dot_products - sums
    denominators = np.array([sums[i] * (sums[i] - 1) for i in subject_indices])
    subject_agreements = np.divide(numerators, denominators)
    return subject_agreements

def observed(data):
    """Computes the observed agreement, P̅."""
    percent_agreement = np.mean(subject_agreements(data))
    return percent_agreement

def expected(data):
    """Computes the expected agreement, P̅(e)."""
    proportions = label_proportions(data)
    percent_expected = np.dot(proportions, proportions)
    return percent_expected