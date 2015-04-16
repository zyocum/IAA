import numpy
from sklearn import metrics
from itertools import permutations

class Annotator(object):
    def __init__(self, label, annotations):
        self.label = label
        self.annotations = numpy.array(annotations)
    
    def __repr__(self):
        return '{} : {}'.format(self.label, repr(self.annotations))

# Annotators A, B, and C
A = Annotator('A', [0, 0, 1, 1])
B = Annotator('B', [0, 0, 0, 1])
C = Annotator('C', [1, 0, 1, 1])

annotators = A, B, C

# Get list of pair-wise permutations of the annotators
pairs = enumerate(permutations(annotators, 2))

# Set up accumulators for Precision, Recall, and F1 scores
ps, rs, fs = [], [], []

# Compute Precision, Recall, and F1 for each pair
for i, pair in  pairs:
    names = [a.label for a in pair]
    annotations = [a.annotations for a in pair]
    cm = metrics.confusion_matrix(*annotations)
    p = metrics.precision_score(*annotations)
    r = metrics.recall_score(*annotations)
    f = metrics.f1_score(*annotations)
    ps.append(p)
    rs.append(r)
    fs.append(f)
    print 'Pair {}'.format(i)
    print '{} (predicted)\nvs.\n{} (reference):'.format(*pair)
    print '\tPrecision: {}'.format(p)
    print '\tRecall: {}'.format(r)
    print '\tF1: {}'.format(f)

# Compute the macro-average scores over all pairs
print 'Macro precision:', numpy.mean(ps)
print 'Macro recall:', numpy.mean(rs)
print 'Macro F1:', numpy.mean(fs)