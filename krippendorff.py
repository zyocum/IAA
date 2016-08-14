# -*- mode: Python; coding: utf-8 -*-
"""Functions for computing Krippendorff's alpha (α), a measure of 
inter-annotator agreement between two or more annotators."""

import csv
import sys
import math
import types
from itertools import permutations

import numpy as np

SENTIMENT_MAP = {
    'positive' : 3,
    'neutral' : 2,
    'negative' : 1,
    'pos' : 3,
    'neu' : 2,
    'neg' : 1
}

def normalize(row, label_map=None):
    """Coerce non-numeric values in a row to None, and the rest to floats"""
    for value in row:
        if label_map:
            value = label_map.get(value)
        try:
            yield float(value)
        except ValueError:
            yield None
        except TypeError:
            yield None

def load(datafile, label_map=None):
    """Load data from a CSV
    
    values must be numerical or empty
    (columns correspond to annotators; rows correspond to subjects)"""
    with open(datafile) as f:
        reader = csv.reader(f)
        raw_data = []
        for row in reader:
            raw_data.append(list(normalize(row, label_map)))
        rows = len(raw_data)
        columns = max(map(len, raw_data))
        data = np.zeros((rows, columns), dtype=object)
        for i in range(rows):
            row = raw_data[i]
            for j in range(columns):
                data[i,j] = row[j] if j < len(row) else None
        return data

def load_stanford_sentiment(datafile):
    """Load data from a CSV
    
    values must be numerical or empty
    (first column is ignored, and remaining columns correspond to annotators;
    rows correspond to subjects)"""
    with open(datafile) as f:
        reader = csv.reader(f)
        raw_data = []
        for row in reader:
            row.pop(0)
            raw_data.append(list(normalize(row)))
        rows = len(raw_data)
        columns = max(map(len, raw_data))
        data = np.zeros((rows, columns), dtype=float)
        for i in range(rows):
            row = raw_data[i]
            for j in range(columns):
                data[i,j] = row[j] if j < len(row) else None
        return data

def nominal_difference(v1, v2, _):
    """Return 0.0 if v1 and v2 share the same value, 1.0 otherwise"""
    return float(v1 != v2)

def metric_difference(v1, v2, values):
    """Return the metric difference between v1 and v2 in the range [0.0,1.0]
    
    The metric difference is normalized by the maximum value in the data set"""
    return np.divide(
        abs(v1 - v2),
        np.max(filter(numeric, values))
    )

def interval_difference(v1, v2, values):
    """Return the interval difference between v1 and v2
    
    The interval difference is the square of the metric difference"""
    return np.square(metric_difference(v1, v2, values))

def get_codebook(data_filter, data):
    """Return a codebook that maps labels/values to indices
    
    The codebook can be used to look up the corresponding column or row
    in a confusion matrix given a value/label"""
    values = set(filter(data_filter, data.flatten()))
    return dict(
        (v,i) for (i,v) in enumerate(values)
    )

def numeric(value):
    """Predicate to deterimine if a given value is a numeric type
    and not 'nan'"""
    numeric_types = (int, float, long, complex)
    is_numeric = any(isinstance(value, t) for t in numeric_types)
    return is_numeric and not math.isnan(value)

def nominal(value):
    """Predicate to determine if a given value can be treated as nominal"""
    return numeric(value) or (isinstance(value, basestring) and value)

def get_coincidence_matrix(data, codebook, data_filter):
    """Given an N x M matrix D (data) with N subjects and M annotators/coders,
    produce an L x L coincidence matrix C where L is the number of labels/values
    assigned in the data such that cell C[i,j] is the frequency that the 
    annotators assigned labels i and j to a subject."""
    labels = set(codebook.keys())
    shape = (len(labels), len(labels))
    matrix = np.zeros(shape, dtype=float)
    for row in data:
        unit = filter(data_filter, row)
        if len(unit) > 1:
            for v1, v2 in permutations(unit, 2):
                i, j = codebook[v1], codebook[v2]
                matrix[i,j] += 1.0 / (len(unit) - 1.0)
    return matrix.astype(int)

def delta(coincidence_matrix, inverse_codebook, difference):
    """Compute a delta vector from a data matrix, a coincidence matrix, and a 
    difference function"""
    delta = []
    for i in range(len(coincidence_matrix)):
        for j in range(i+1, len(coincidence_matrix)):
            v1, v2 = inverse_codebook[i], inverse_codebook[j]
            delta.append(difference(v1, v2, inverse_codebook.values()))
    return np.array(delta)

def observation(coincidence_matrix, codebook, d):
    """Compute the observed agreement D(o)"""
    o = []
    for i in range(len(coincidence_matrix)):
        for j in range(i+1, len(coincidence_matrix)):
            o.append(coincidence_matrix[i,j])
    observation = np.dot(o, d)
    return observation

def expectation(coincidence_matrix, d):
    """Compute the expected agreement D(e)"""
    n = []
    for i in range(len(coincidence_matrix)):
        for j in range(i+1, len(coincidence_matrix)):
            cm_i = np.sum(coincidence_matrix[i])
            cm_j = np.sum(coincidence_matrix[j])
            n.append(cm_i * cm_j)
    expectation = np.divide(
        np.dot(n, d),
        sum(coincidence_matrix.sum(axis=1)) - 1
    )
    return expectation

def krippendorff(data, difference):
    """Compute Krippendorff's α given a matrix D (data) of annotations
    
    The data matrix D must be an N x M matrix where N is the number of subjects
    and M is the number of annotators/coders."""
    if not type(data) == np.ndarray:
        raise TypeError, 'expected a numpy array'
    if not type(difference) == types.FunctionType:
        raise TypeError, 'expected a difference function'
    if len(data.shape) != 2:
        raise ValueError, 'input must be 2-dimensional array'
    if difference.__name__ == 'nominal_difference':
        data_filter = nominal
    else:
        data_filter = numeric
    cb = get_codebook(data_filter, data)
    icb = dict((i,v) for (v,i) in cb.iteritems())
    cm = get_coincidence_matrix(data, cb, data_filter)
    d = delta(cm, icb, difference)
    observed = observation(cm, cb, d)
    expected = expectation(cm, d)
    perfection = 1.0
    a = perfection - np.divide(observed, expected)
    return a

if __name__ == '__main__':
    filename = sys.argv[1]
    data = load_stanford_sentiment(filename)
    #data = load(filename, ENTITY_TYPES_MAP)
    dfs = [
        nominal_difference,
        metric_difference,
        interval_difference
    ]
    for df in dfs:
        print '{} : {}'.format(df.__name__, krippendorff(data, df))
