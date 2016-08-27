#!/usr/bin/env python
# -*- mode: Python; coding: utf-8 -*-
"""Functions for computing Krippendorff's alpha (α), a measure of 
inter-annotator agreement between two or more annotators."""

import csv
import sys
import math
import types
from warnings import warn
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
        """Load data from TSV (rows = subjects; columns = annotators)"""
        if isinstance(datafile, basestring) and os.path.isfile(datafile):
            with open(datafile, mode='r') as f:
                reader = csv.reader(f, delimiter='\t')
        elif isinstance(datafile, file):
            reader = csv.reader(datafile, delimiter='\t')
        else:
            raise ValueError, 'datafile must be an open file, or a file path'
        raw_data = []
        for row in reader:
            raw_data.append(map(self.get, row))
        if not raw_data:
            raise ValueError, 'input data must be non-empty'
        rows = len(raw_data)
        columns = max(map(len, raw_data))
        data = np.zeros((rows, columns), dtype=self.type)
        for i in range(rows):
            row = raw_data[i]
            for j in range(columns):
                data[i,j] = row[j] if j < len(row) else None
        return data

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

class Ordinal(DataType):
    """A numeric data type with a metric difference function"""
    def __init__(self):
        self.type = float
    
    def __getitem__(self, arg):
        try:
            number = self.type(arg)
        except ValueError:
            return None
        if any(isinstance(number, t) for t in NUMBERS):
            if np.isfinite(number):
                return number
            else:
                return None
    
    def difference(self, v1, v2, values):
        """Return the metric difference between v1 and v2 between [0.0,1.0]
    
        The difference is normalized by the maximum value in the data set"""
        return np.divide(abs(v1 - v2), np.max(values))

class Interval(Ordinal):
    """A numeric data type with an interval difference function"""
    
    def difference(self, v1, v2, values):
        """Return the interval difference between v1 and v2 between [0.0,1.0]
    
        The difference is normalized by the maximum value in the data set"""
        return np.divide((v1 - v2) ** 2, np.max(values))

def get_coincidence_matrix(data, codebook, data_type):
    """Return a coincidence matrix
    
    Given an N x M matrix D (data) with N subjects and M annotators/coders,
    produce an L x L coincidence matrix C where L is the number of labels/values
    assigned in the data such that cell C[i,j] is the frequency that the 
    annotators assigned labels i and j to a subject."""
    labels = set(codebook.keys())
    shape = (len(labels), len(labels))
    matrix = np.zeros(shape, dtype=float)
    for row in data:
        unit = [x for x in map(data_type.get, row) if x is not None]
        if len(unit) > 1:
            for v1, v2 in permutations(unit, 2):
                i, j = codebook[v1], codebook[v2]
                matrix[i,j] += 1.0 / (len(unit) - 1.0)
    return matrix

def delta(coincidence_matrix, inverse_codebook, difference):
    """Compute a delta vector.
    
    Given a coincidence matrix, an inverse codebook, and a difference function,
    compute a delta vector that can be used to compute observed and expected
    agreement.  The inverse codebook allows for looking up values from the
    row/column indices of the coincidence matrix."""
    delta = []
    for i in range(len(coincidence_matrix)):
        for j in range(i+1, len(coincidence_matrix)):
            v1, v2 = inverse_codebook[i], inverse_codebook[j]
            delta.append(difference(v1, v2, inverse_codebook.values()))
    return np.array(delta)

def observation(coincidence_matrix, d):
    """Compute the observed agreement D(o).
                    
    D(o) = 𝛴[v=1,v'=1 → V] o(v,v') * δ(v,v')
    
    Where...
              V = the size of the set of values/labels that occur in the data
              v = a row index of the coincidence matrix
             v' = a column indix of the coincidence matrix
        o(v,v') = the frequency in cell C[v,v'] in the coincidence matrix C
        δ(v,v') = the difference function applied to the values of v and v'
    """
    o = []
    for i in range(len(coincidence_matrix)):
        for j in range(i+1, len(coincidence_matrix)):
            o.append(coincidence_matrix[i,j])
    observation = np.dot(o, d)
    return observation

def expectation(coincidence_matrix, d):
    """Compute the expected agreement D(e).
    
            𝛴[v=1,v'=1 → V] n(v) * n(v') * δ(v,v')
    D(e) = ----------------------------------------
                            n - 1
    
    Where...
              V = the size of the set of values/labels that occur in the data
              v = a row index of the coincidence matrix
             v' = a column indix of the coincidence matrix
           n(v) = the sum of the row v of the coincidence matrix
          n(v') = the sum of the column v' of the coincidence matrix
        δ(v,v') = the difference function applied to the values of v and v'
              n = the sum of the coincidence matrix
    """
    n = []
    for i in range(len(coincidence_matrix)):
        for j in range(i+1, len(coincidence_matrix)):
            cm_i = np.sum(coincidence_matrix[i])
            cm_j = np.sum(coincidence_matrix[j])
            n.append(cm_i * cm_j)
    expectation = np.divide(np.dot(n, d), coincidence_matrix.sum() - 1)
    return expectation

def krippendorff(data, data_type):
    """Compute Krippendorff's α.
    
    Given a matrix D (data) of annotations and a data type class with a
    difference function δ, compute the agreement score.  The data matrix D must
    be an N x M matrix where N is the number of subjects and M is the number of
    annotators.
    
    From the data matrix D a coincidence matrix C is computed with dimensions
    V x V where V is the size of the set of values assigned in the data. Thus,
    each column and row of the coincidence matrix C corresponds to a value/label
    in the data, and the frequency with which the pair of labels v and v' were
    assigned are stored in cell C[v,v'].  The diagonal of C contains the
    frequencies of instances where annotators agreed, and values above and below
    the diagonal are symmetric, containing the frequencies of disagreements.
    
             D(o)         (n - 1) * 𝛴[v=1,v'=1 → V] o(v,v') * δ(v,v')
    α = 1 - ------ = 1 - ---------------------------------------------
             D(e)           𝛴[v=1,v'=1 → V] n(v) * n(v') * δ(v,v')
    
    Where...
           D(o) = the observed agreement
           D(e) = the expected agreement
              n = the sum of the coincidence matrix
              v = a row index of the coincidence matrix
             v' = a column index of the coincidence matrix
              V = the size of the set of values/labels that occur in the data
        o(v,v') = the value in cell [v,v'] in the coincidence matrix
        δ(v,v') = the difference function applied to the values of v and v'
           n(v) = the sum of the row v of the coincidence matrix
          n(v') = the sum of the column v' of the coincidence matrix
    """
    if not type(data) == np.ndarray:
        raise TypeError, 'expected a numpy array'
    if len(data.shape) != 2:
        raise ValueError, 'input must be 2-dimensional array'
    if data.shape < (1, 2):
        message = 'input must be at least 1 row x 2 columns (rows x annotators)'
        raise ValueError, message
    values = set(map(data_type.get, data.flatten())) - {None}
    if not any(values):
        message = 'input must include at least one value/label of type {}'
        raise ValueError, message.format(data_type.type.__name__)
    if len(values) == 1:
        print >> sys.stderr, 'Warning: all input values are identical!'
        return 1.0
    codebook = dict((v,i) for (i,v) in enumerate(values))
    inverse_codebook = dict(enumerate(values))
    cm = get_coincidence_matrix(data, codebook, data_type)
    d = delta(cm, inverse_codebook, data_type.difference)
    observed = observation(cm, d)
    expected = expectation(cm, d)
    perfection = 1.0
    a = perfection - np.divide(observed, expected)
    return a

if __name__ == '__main__':
    import argparse
    import os
    DATA_TYPES = (Nominal, Ordinal, Interval)
    DATA_TYPES_DICT = dict((dt().name().lower(), dt()) for dt in DATA_TYPES)
    parser = argparse.ArgumentParser(
        description="Compute Krippendorff's alpha (α), a measure of \
        interannotator agreement between two or more annotators."
    )
    parser.add_argument(
        '-i',
        '--input',
        default=sys.stdin,
        metavar='data.csv',
        help='''a TSV data file (rows=subjects, columns=annotators)
        (data is read from stdin if no path is given)'''
    )
    parser.add_argument(
        '-t',
        '--type',
        default='nominal',
        choices=DATA_TYPES_DICT.keys(),
        help='how to treat the data labels (default="nominal")'
    )
    args = parser.parse_args()
    data_type = DATA_TYPES_DICT[args.type.lower()]
    data = data_type.load(args.input)
    print 'δ: {} difference'.format(data_type.name())
    print 'α: {}'.format(krippendorff(data, data_type))
