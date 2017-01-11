#!/usr/bin/env python3
# -*- mode: Python; coding: utf-8 -*-
"""Test suite for inter-annotator agreement kappa coefficients."""

import numpy as np
import unittest, cohen, fleiss

class KappaExample(object):
    """A class to facilitate printing kappa examples."""
    def __init__(self, kappa, data, separator=' : '):
        self.kappa = kappa
        self.data = np.array(data)
        self.separator = separator
        self.title = "{}'s kappa".format(self.kappa.__name__.title())
        self.labels = (
            'Data',
            'Observed agreement',
            'Expected agreement',
            'Kappa score'
        )
        self.observed = self.kappa.observed(self.data)
        self.expected = self.kappa.expected(self.data)
        self.score = self.kappa.kappa(self.data)
    
    def __repr__(self):
        width = max(map(len, self.labels))
        indent = ' ' * (width + len(self.separator))
        matrix = ('\n' + indent).join(self.prettify(self.data).split('\n'))
        line = '{{:>{w}}}{sep}{{}}'.format(w=width, sep=self.separator)
        values = (
            matrix,
            self.observed,
            self.expected,
            self.score
        )
        records = zip(self.labels, values)
        lines = [self.title] + [line.format(*record) for record in records]
        return '\n'.join(lines)
    
    @staticmethod
    def prettify(data, rowsep=(',\n       ', '\n'), colsep=(', ', ' ')):
        """Get a cleaner/prettier representation of a 2-d numpy.array."""
        cluttered = repr(data)
        clean = cluttered.strip('array()')[1:-1]
        pretty = clean.replace(*rowsep).replace(*colsep)
        return pretty

# Cohen's 𝜅 example data from:
# Pustejovsky and Stubbs
# Natural Language Annotation for Machine Learning, (p. 127-128)
# This example is for 2 annotators (A and B) who have each labeled 250 
# documents in terms of their sentiment as positive (+), neutral (/) or
# negative (-).
COHEN_EXAMPLE = KappaExample(
    cohen,
    #  B   B   B
    #  +   /   -
    [[54, 28,  3], # + A
     [31, 18, 23], # / A
     [ 0, 21, 72]] # - A
)
# Pr(a) ~= 0.338608
# Pr(e) ~= 0.576
# Cohen's 𝜅 ~= 0.358927837047

# Fleiss's 𝜅 example data from:
# https://en.wikipedia.org/wiki/Fleiss%27_kappa
# This example is for 14 annotators who have assigned 5 labels to 10
# subjects.
FLEISS_EXAMPLE = KappaExample(
    fleiss,
    #     labels
    #  0  1  2  3   4
    [[ 0, 0, 0, 0, 14], #0
     [ 0, 2, 6, 4,  2], #1 s
     [ 0, 0, 3, 5,  6], #2 u
     [ 0, 3, 9, 2,  0], #3 b
     [ 2, 2, 8, 1,  1], #4 j
     [ 7, 7, 0, 0,  0], #5 e
     [ 3, 2, 6, 3,  0], #6 c
     [ 2, 5, 3, 2,  2], #7 t
     [ 6, 5, 2, 1,  0], #8 s
     [ 0, 2, 2, 3,  7]] #9
)
# P̅ ~= 0.212755102041
# P̅(e) ~= 0.378021978022
# Fleiss's 𝜅 ~= 0.209930704422

EXAMPLES = COHEN_EXAMPLE, FLEISS_EXAMPLE

class CohenTestCase(unittest.TestCase):
    def setUp(self):
        self.example = COHEN_EXAMPLE
    
    def test_expected(self):
        self.assertAlmostEqual(self.example.expected, 0.338608)
    
    def test_observed(self):
        self.assertAlmostEqual(self.example.observed, 0.576)
    
    def test_kappa(self):
        self.assertAlmostEqual(self.example.score, 0.358927837047)

class FleissTestCase(unittest.TestCase):
    def setUp(self):
        self.example = FLEISS_EXAMPLE
    
    def test_expected(self):
        self.assertAlmostEqual(self.example.expected, 0.212755102041)
    
    def test_observed(self):
        self.assertAlmostEqual(self.example.observed, 0.378021978022)
    
    def test_kappa(self):
        self.assertAlmostEqual(self.example.score, 0.209930704422)

if __name__ == '__main__':
    for example in EXAMPLES:
        print(example)
    # Test suite
    def suite():
        suite = unittest.TestSuite()
        cases = (CohenTestCase, FleissTestCase)
        tests = ('test_expected', 'test_observed', 'test_kappa')
        for case in cases:
            for test in tests:
                suite.addTest(case(test))
        return suite
    unittest.main(verbosity=2)