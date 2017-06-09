#!/usr/bin/env python3
# -*- mode: Python; coding: utf-8 -*-
"""Functions for computing Krippendorff's alpha (α), a measure of 
inter-annotator agreement between two or more annotators."""

import sys
from itertools import permutations
from operator import attrgetter, methodcaller

import numpy as np
import pandas as pd

NUMBERS = int, float, complex

DEFAULT_NAN_VALUES = [
    '',
    '#N/A',
    '#N/A N/A',
    '#NA',
    '-1.#IND',
    '-1.#QNAN',
    '-NaN',
    '-nan',
    '1.#IND',
    '1.#QNAN',
    'N/A',
    'NA',
    'NULL',
    'NaN',
    'nan'
]

DATA_TYPES = 'complex', 'double', 'float', 'int', 'str', 'object_'

def load(datafile, **kwargs):
    """Load data from file via pandas.DataFrame and convert to numpy.array"""
    raw = pd.read_table(datafile, **kwargs)
    if raw.empty:
        raise ValueError('input data must be non-empty')
    dtypes = set(raw.dtypes)
    if len(dtypes) != 1:
        message = 'input data must be uniformly typed (found {} types: {})'
        raise ValueError(message.format(len(dtypes), dtypes))
    data = raw.as_matrix()
    return data

class Difference():
    def __init__(self, dtype, method='nominal'):
        self.dtype = dtype
        self.method = method
    
    def nominal(self, v1, v2, *args):
        """Return 0.0 if v1 and v2 are the same value, 1.0 otherwise"""
        return float(v1 != v2)
    
    def metric(self, v1, v2, values, *args):
        """Return the metric difference between v1 and v2 between [0.0,1.0]
        
        The difference is normalized by the maximum value in the data set
        """
        return np.divide(abs(v1 - v2), abs(np.max(values) - np.min(values)))
    
    def ordinal(self, v1, v2, *args):
        """Return the ordinal difference between ranks v1 and v2"""
        # force ordinal range to span from lowest to highest value
        v1, v2 = sorted((v1, v2))
        return (np.sum(np.arange(v1, v2 + 1)) - np.divide(v1 + v2, 2)) ** 2
    
    def interval(self, v1, v2, values, *args):
        """Return the interval difference between v1 and v2
        
        The difference is normalized by the maximum value in the data set
        """
        return np.divide((v1 - v2) ** 2, np.max(values))
    
    def _delta(self, *args):
        """Convenience method for calling Difference.method(*args)"""
        return methodcaller(self.method, *args)(self)

def get_coincidence_matrix(data, codebook):
    """Return a coincidence matrix
    
    Given an N x M matrix D (data) with N subjects and M annotators/coders,
    produce an L x L coincidence matrix C where L is the number of labels/values
    assigned in the data such that cell C[i,j] is the frequency that the 
    annotators assigned labels i and j to a subject."""
    labels = set(codebook.keys())
    shape = (len(labels), len(labels))
    matrix = np.zeros(shape, dtype=data.dtype)
    for row in data:
        unit = [x for x in row if x == x]
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
            values = list(inverse_codebook.values())
            delta.append(difference._delta(v1, v2, values))
    return np.array(delta)

def observation(coincidence_matrix, d):
    """Compute the observed agreement D(o).
                    
    D(o) = 𝛴(v=1,v'=1 → V)[o(v,v') * δ(v,v')]
    
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
    
            𝛴(v=1,v'=1 → V)[n(v) * n(v') * δ(v,v')]
    D(e) = -----------------------------------------
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

def alpha(data, difference):
    """Compute Krippendorff's α.
    
    Given a matrix D (data) of annotations and a difference class with a
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
    
             D(o)         (n - 1) * 𝛴(v=1,v'=1 → V)[o(v,v') * δ(v,v')]
    α = 1 - ------ = 1 - ----------------------------------------------
             D(e)           𝛴(v=1,v'=1 → V)[n(v) * n(v') * δ(v,v')]
    
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
    if isinstance(data, pd.DataFrame):
        data = data.as_matrix()
    if not isinstance(data, np.ndarray):
        raise TypeError('expected a pandas.DataFrame or np.ndarray')
    if data.ndim != 2:
        raise ValueError('input must be 2-dimensional array')
    if data.shape < (1, 2):
        message = 'input must have dimensions at least 1 x 2 (rows x columns)'
        raise ValueError(message)
    values = set(v for v in data.flatten() if v == v)
    if not np.any(values):
        message = 'input must include at least one value/label'
        raise ValueError(message)
    if len(values) == 1:
        print('Warning: all input values are identical!', file=sys.stderr)
        return 1.0
    codebook = {v : i for (i, v) in enumerate(values)}
    inverse_codebook = dict(enumerate(values))
    cm = get_coincidence_matrix(data, codebook)
    d = delta(cm, inverse_codebook, difference)
    observed = observation(cm, d)
    expected = expectation(cm, d)
    perfection = 1.0
    a = perfection - np.divide(observed, expected)
    return a

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute Krippendorff's alpha (α), a measure of \
        interannotator agreement between two or more annotators.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i',
        '--input',
        default=sys.stdin,
        metavar='data.csv',
        help="A data file (rows=subjects, columns=annotators)"
    )
    parser.add_argument(
        '-t',
        '--dtype',
        default='object_',
        choices=DATA_TYPES,
        help='The type of data to load'
    )
    parser.add_argument(
        '-f',
        '--difference',
        '--delta',
        default='nominal',
        choices=[m for m in dir(Difference) if not m.startswith('_')],
        help='The difference method (delta or δ) to use'
    )
    parser.add_argument(
        '-d',
        '--delimiter',
        '--sep',
        default="\t",
        type=str,
        help=(
            'Delimiter to use. If sep is None, will try to automatically '
            'determine this. Separators longer than 1 character and different '
            "from ``'\s+'`` will be interpreted as regular expressions, will "
            'force use of the python parsing engine and will ignore quotes in '
            "the data. Regex example: ``'\r\t'``"
        )
    )
    parser.add_argument(
        '--na-values',
        '--nil-values',
        '--null-values',
        default=DEFAULT_NAN_VALUES,
        nargs='+',
        help='List of strings to recognize as NA/NaN/null.',
    )
    parser.add_argument(
        '--skip-blank-lines',
        action='store_true',
        help='Skip over blank lines rather than interpreting as NaN values'
    )
    parser.add_argument(
        '--no-header',
        action='store_true',
        help='Indicate that there is no header row in the data'
    )
    parser.add_argument(
        '-c',
        '--usecols',
        '--columns',
        nargs='+',
        metavar='COLUMN',
        default=None,
        help=(
            'A subset of the columns. All elements in this list must either '
            'be positional (i.e. integer indices into the columns) or strings '
            'that correspond to column names provided either by the user in '
            '`names` or inferred from the document header row(s). For example, '
            "a valid `usecols` parameter would be [0, 1, 2] or ['foo', 'bar', "
            "'baz']. Using this parameter results in much faster parsing "
            'time and lower memory usage.'
        )
    )
    parser.add_argument(
        '-n',
        '--names',
        nargs='+',
        default=None,
        help=(
            'List of column names to use. If file contains no header row, then '
            'you should explicitly use --no-header. Duplicates in this list '
            'are not allowed unless mangle_dupe_cols=True, which is the '
            'default.'
        )
    )
    args = parser.parse_args()
    dtype = attrgetter(args.dtype)(np)
    nan_values = args.na_values
    try:
        data = load(
            args.input,
            header=None if args.no_header else 'infer',
            names=args.names,
            delimiter=args.delimiter,
            na_values=args.na_values,
            skip_blank_lines=args.skip_blank_lines,
            usecols=args.usecols,
            dtype=args.dtype
        )
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    difference = Difference(dtype, args.difference)
    print('δ: {} difference'.format(args.difference))
    print('α: {}'.format(alpha(data, difference)))
