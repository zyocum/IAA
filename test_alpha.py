#!/usr/bin/env python3
# -*- mode: Python; coding: utf-8 -*-
"""Test suite for Krippendorff's alpha coefficient.

    Example matrix D:
    |-------------+----+---+---+----|
    |  Annotators |    | A | B | C  |
    |-------------+----+---+---+----|
    |  Subject    | 1  |   | 1 |    |
    |  Subject    | 2  |   |   |    |
    |  Subject    | 3  |   | 2 | 2  |
    |  Subject    | 4  |   | 1 | 1  |
    |  Subject    | 5  |   | 3 | 3  |
    |  Subject    | 6  | 3 | 3 | 4  |
    |  Subject    | 7  | 4 | 4 | 4  |
    |  Subject    | 8  | 1 | 3 |    |
    |  Subject    | 9  | 2 |   | 2  |
    |  Subject    | 10 | 1 |   | 1  |
    |  Subject    | 11 | 1 |   | 1  |
    |  Subject    | 12 | 3 |   | 3  |
    |  Subject    | 13 | 3 |   | 3  |
    |  Subject    | 14 |   |   |    |
    |  Subject    | 15 | 3 |   | 4  |
    |-------------+----+---+---+----|
    
    Example coincidence matrix C (computed from D):
    |-----------------+---+---+---+----+---+------|
    |  Values v'      |   | 1 | 2 | 3  | 4 | n(v) |
    |-----------------+---+---+---+----+---+------|
    |  Value v        | 1 | 6 |   | 1  |   | 7    |
    |  Value v        | 2 |   | 4 |    |   | 4    |
    |  Value v        | 3 | 1 |   | 7  | 2 | 10   |
    |  Value v        | 4 |   |   | 2  | 3 | 5    |
    |-----------------+---+---+---+----+---+------|
    |  Frequency n(v)'|   | 7 | 4 | 10 | 5 | 26   |
    |-----------------+---+---+---+----+---+------|
    
                                      1 + 2
                     ---------------------------------------- 
    α(nominal) = 1 -  (4*7 + 10*7 + 5*7 + 10*4 + 5*4 + 5*10)   ≈ 0.691
                                     --------  
                                      26 - 1
    
                                       1 + 2
                      ------------------------------------------ 
    α(metric) = 1 -   (4*7*1+10*7*2+5*7*3+10*4*1+5*4*2+5*10*1)   ≈ 0.751
                                      --------  
                                       26 - 1
    
                                       1 * 2² + 2 * 1²
                      ------------------------------------------------ 
    α(interval) = 1 -  (4*7*1²+10*7*2²+5*7*3²+10*4*1²+5*4*2²+5*10*1²)   ≈ 0.811
                                          --------  
                                           26 - 1

./krippendorff.py -f nominal -n {1..6} --no-header < sample-data/stanford_sentiment_raw.tsv
δ: nominal difference
α: 0.18663461373179746

./krippendorff.py -t float -f metric -n {1..6} --no-header < sample-data/stanford_sentiment_raw.tsv
δ: metric difference
α: 0.4494141275145108

./krippendorff.py -t float -f ordinal -n {1..6} --no-header < sample-data/stanford_sentiment_raw.tsv
δ: ordinal difference
α: 0.5756256865163973

./krippendorff.py -t int -f ordinal -n {1..6} --no-header < sample-data/stanford_sentiment_raw.tsv
Integer column has NA values in column 3

./krippendorff.py -t float -f interval -n {1..6} --no-header < sample-data/stanford_sentiment_raw.tsv
δ: interval difference
α: 0.5935814573071891

./krippendorff.py -f nominal --no-header < sample-data/sentiment_nominal.tsv
δ: nominal difference
α: 0.3600164551295236

./krippendorff.py -t float -f metric --no-header < sample-data/sentiment_numeric.tsv
δ: metric difference
α: 0.5275236928743301

./krippendorff.py -t float -f ordinal --no-header < sample-data/sentiment_numeric.tsv
δ: ordinal difference
α: 0.6921019808933906

./krippendorff.py -t int -f ordinal --no-header < sample-data/sentiment_numeric.tsv
δ: ordinal difference
α: 0.6921019808933906

./krippendorff.py -t float -f interval --no-header < sample-data/sentiment_numeric.tsv
δ: interval difference
α: 0.6812989075802931

./krippendorff.py -f nominal < sample-data/test.txt
δ: nominal difference
α: 0.691358024691358

./krippendorff.py -t float -f metric < sample-data/test.txt
δ: metric difference
α: 0.7518610421836228

./krippendorff.py -t float -f ordinal < sample-data/test.txt
δ: ordinal difference
α: 0.786132967207055

./krippendorff.py -t int -f ordinal < sample-data/test.txt
Integer column has NA values in column 0

./krippendorff.py -t float -f interval < sample-data/test.txt
δ: interval difference
α: 0.8108448928121059

"""

import numpy as np
import unittest, krippendorff

class AlphaExample():
    """A class to facilitate printing alpha examples."""
    def __init__(self, separator=': ', **kwargs):
        self.target = kwargs
        self.data = np.array(kwargs['data'])
        self.difference = krippendorff.Difference(*kwargs['args'])
        self.separator = separator
        self.labels = (
            'Data',
            'Data type',
            'Difference method',
            'Observed agreement',
            'Expected agreement',
            'Alpha score'
        )
        self.values = set(v for v in self.data.flatten() if v == v)
        self.codebook = {v : i for (i, v) in enumerate(self.values)}
        self.inverse_codebook = dict(enumerate(self.values))
        self.cm = krippendorff.get_coincidence_matrix(self.data, self.codebook)
        self.d = krippendorff.delta(
            self.cm,
            self.inverse_codebook,
            self.difference
        )
        self.observed = krippendorff.observation(self.cm, self.d)
        self.expected = krippendorff.expectation(self.cm, self.d)
        self.alpha = krippendorff.alpha(self.data, self.difference)
    
    def __repr__(self):
        width = max(map(len, self.labels))
        indent = ' ' * (width + len(self.separator))
        matrix = ('\n' + indent).join(self.prettify(self.data).split('\n'))
        line = '{{:>{w}}}{sep}{{}}'.format(w=width, sep=self.separator)
        values = (
            matrix,
            self.difference.dtype.__name__,
            self.difference.method,
            self.observed,
            self.expected,
            self.alpha
        )
        records = zip(self.labels, values)
        lines = [line.format(*record) for record in records]
        return '\n'.join(lines)
    
    @staticmethod
    def prettify(data, rowsep=(',\n       ', '\n'), colsep=(', ', ' ')):
        """Get a cleaner/prettier representation of a 2-d numpy.array."""
        cluttered = repr(data)
        clean = cluttered.strip('array()')[1:-1]
        pretty = clean.replace(*rowsep).replace(*colsep)
        return pretty

Nan = float('nan')

DATA = [
    [Nan,   1, Nan],
    [Nan, Nan, Nan],
    [Nan,   2,   2],
    [Nan,   1,   1],
    [Nan,   3,   3],
    [3,     3,   4],
    [4,     4,   4],
    [1,     3, Nan],
    [2,   Nan,   2],
    [1,   Nan,   1],
    [1,   Nan,   1],
    [3,   Nan,   3],
    [3,   Nan,   3],
    [Nan, Nan, Nan],
    [3,    Nan,  4]
]

EXAMPLES = [
    {
        'data': DATA,
        'args': (np.str, 'nominal'),
        'observed': 3.0,
        'expected': 9.72,
        'alpha': 0.69135802469135799
    },
    {
        'data': DATA,
        'args': (np.int, 'nominal'),
        'observed': 3.0,
        'expected': 9.72,
        'alpha': 0.69135802469135799
    },
    {
        'data': DATA,
        'args': (np.float, 'nominal'),
        'observed': 3.0,
        'expected': 9.72,
        'alpha': 0.69135802469135799
    },
    {
        'data': DATA,
        'args': (np.float, 'metric'),
        'observed': 1.3333333333333333,
        'expected': 5.3733333333333322,
        'alpha': 0.7518610421836228
    },
    {
        'data': DATA,
        'args': (np.int, 'metric'),
        'observed': 1.3333333333333333,
        'expected': 5.3733333333333322,
        'alpha': 0.7518610421836228
    },
    {
        'data': DATA,
        'args': (np.int, 'ordinal'),
        'observed': 40.5,
        'expected': 189.37,
        'alpha': 0.78613296720705494
    },
    {
        'data': DATA,
        'args': (np.float, 'ordinal'),
        'observed': 40.5,
        'expected': 189.37,
        'alpha': 0.78613296720705494
    },
    {
        'data': DATA,
        'args': (np.int, 'interval'),
        'observed': 1.5,
        'expected': 7.9299999999999997,
        'alpha': 0.81084489281210592
    },
    {
        'data': DATA,
        'args': (np.float, 'interval'),
        'observed': 1.5,
        'expected': 7.9299999999999997,
        'alpha': 0.81084489281210592
    },
    
]

EXAMPLES = [AlphaExample(**args) for args in EXAMPLES]

class AlphaTestCase(unittest.TestCase):
    def setUp(self):
        self.examples = EXAMPLES
        
    def test_expected(self):
        for example in self.examples:
            self.assertAlmostEqual(example.target['expected'], example.expected)
    
    def test_observed(self):
        for example in self.examples:
            self.assertAlmostEqual(example.target['observed'], example.observed)
    
    def test_alpha(self):
        for example in self.examples:
            self.assertAlmostEqual(example.target['alpha'], example.alpha)

if __name__ == '__main__':
    # Test suite
    def suite():
        suite = unittest.TestSuite()
        tests = ('test_expected', 'test_observed', 'test_alpha')
        for test in tests:
            suite.addTest(AlphaTestCase(test))
        return suite
    
    unittest.main(verbosity=2)
