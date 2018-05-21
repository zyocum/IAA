#!/usr/bin/env python3

""""Compute a pair-wise TSV matrix of Krippendorff's alpha (α) scores

(Only pairs above the diagonal are computed.  The cells on the diagonal of the
pair-wise matrix are trivally 1.0 since every annotator agrees completely with
his/herself and values below the diagonal would be redundant with the 
symmetrical cell above the diagonal.)"""

import sys
import numpy as np
import pandas as pd

from itertools import combinations
from operator import attrgetter, methodcaller

from krippendorff import alpha, Difference, load, DEFAULT_NAN_VALUES, DATA_TYPES

def main(data, difference):
    """From a data matrix/data frame and a given difference method, compute
    a pair-wise matrix such that the cell at matrix[x,y] is the alpha score for
    the pair of annotators x and y.
    
    E.g., if there are 5 annotators corresponding to 5 columns in data, then 
    the output matrix[0,1] would be the alpha score between the first two 
    annotators and matrix[1,4] would be the alpha score between the second and 
    fifth annotators."""
    rows, cols = data.shape
    matrix = np.zeros((cols, cols), dtype=float)
    pairs = combinations(range(cols), 2)
    for pair in pairs:
        ann1, ann2 = pair
        matrix[ann1, ann2] = alpha(data[:, [ann1, ann2]], difference)
    return matrix
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__,
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
    matrix = main(data, difference)
    np.savetxt(sys.stdout.buffer, matrix, fmt='%.2f', delimiter='\t', newline='\n')
