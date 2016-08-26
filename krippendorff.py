# -*- mode: Python; coding: utf-8 -*-
"""Functions for computing Krippendorff's alpha (α), a measure of 
inter-annotator agreement between two or more annotators."""

import csv
import sys
import math
import types
from itertools import permutations

import numpy as np

NUMBERS = (int, float, long, complex)

class DataType():
    """Base class for defining data types"""
    def __init__(self, t):
        self.type = t
    
    def __getitem__(self, arg):
        return self.type(arg)
    
    def name(self):
        return self.__class__.__name__
    
    def get(self, arg):
        return self[arg]
    
    def load(self, datafile):
        """Load data from TSV
        (rows = subjects; columns = annotators)"""
        with open(datafile) as f:
            reader = csv.reader(f, delimiter='\t')
            raw_data = []
            for row in reader:
                raw_data.append(map(self.get, row))
            rows = len(raw_data)
            columns = max(map(len, raw_data))
            data = np.zeros((rows, columns), dtype=self.type)
            for i in range(rows):
                row = raw_data[i]
                for j in range(columns):
                    data[i,j] = row[j] if j < len(row) else None
            return data
    
class Numeric(DataType):
    """A numeric data type with a metric difference function"""
    def __init__(self):
        self.type = float
    
    def __getitem__(self, arg):
        try:
            number = self.type(arg)
        except ValueError:
            return None
        if any(isinstance(number, t) for t in NUMBERS):
            if not math.isnan(number):
                return number
            else:
                return None
    
    def difference(self, v1, v2, values):
        """Return the metric difference between v1 and v2 in the range [0.0,1.0]
    
        The difference is normalized by the maximum value in the data set"""
        maximum_value = np.max(filter(None, map(self.get, values)))
        return np.divide(abs(v1 - v2), maximum_value)

class Nominal(DataType):
    """A nominal data type with a nominal difference function"""
    def __init__(self):
        self.type = basestring
    
    def __getitem__(self, arg):
        if isinstance(arg, self.type):
            if arg == '':
                return None
            else:
                return arg
        elif any(isinstance(arg, t) for t in NUMBERS):
            return str(arg)
        else:
            return None
    
    def difference(self, v1, v2, _):
        """Return 0.0 if v1 and v2 are the same value, 1.0 otherwise"""
        return float(v1 != v2)

def get_coincidence_matrix(data, codebook, data_type):
    """Given an N x M matrix D (data) with N subjects and M annotators/coders,
    produce an L x L coincidence matrix C where L is the number of labels/values
    assigned in the data such that cell C[i,j] is the frequency that the 
    annotators assigned labels i and j to a subject."""
    labels = set(codebook.keys())
    shape = (len(labels), len(labels))
    matrix = np.zeros(shape, dtype=float)
    for row in data:
        unit = filter(None, map(data_type.get, row))
        if len(unit) > 1:
            for v1, v2 in permutations(unit, 2):
                i, j = codebook[v1], codebook[v2]
                matrix[i,j] += 1.0 / (len(unit) - 1.0)
    return matrix.astype(int)

def delta(coincidence_matrix, inverse_codebook, difference):
    """Compute a delta vector from a coincidence matrix, an inverse codebook,
    and a difference function
    
    The inverse codebook allows for looking up values from the row/column
    indices of the coincidence matrix"""
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

def krippendorff(data, data_type):
    """Compute Krippendorff's α given a matrix D (data) of annotations
    and a difference function δ.
    
    The data matrix D must be an N x M matrix where N is the number of subjects
    and M is the number of annotators/coders."""
    if not type(data) == np.ndarray:
        raise TypeError, 'expected a numpy array'
    if len(data.shape) != 2:
        raise ValueError, 'input must be 2-dimensional array'
    values = set(filter(None, map(data_type.get, data.flatten())))
    codebook = dict((v,i) for (i,v) in enumerate(values))
    inverse_codebook = dict(enumerate(values))
    cm = get_coincidence_matrix(data, codebook, data_type)
    d = delta(cm, inverse_codebook, data_type.difference)
    observed = observation(cm, codebook, d)
    expected = expectation(cm, d)
    perfection = 1.0
    a = perfection - np.divide(observed, expected)
    return a

if __name__ == '__main__':
    import argparse
    
    DATA_TYPES = dict((dt().name().lower(), dt()) for dt in (Nominal, Numeric))
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        'filename',
        metavar='data.csv',
        help='a CSV data file (rows=subjects, columns=annotators)'
    )
    parser.add_argument(
        '-t',
        '--type',
        default='nominal',
        choices=DATA_TYPES.keys(),
        help='how to treat the data labels (default="nominal")'
    )
    args = parser.parse_args()
    data_type = DATA_TYPES[args.type.lower()]
    data = data_type.load(args.filename)
    print 'δ: {} difference'.format(data_type.name())
    print 'α: {}'.format(krippendorff(data, data_type))
