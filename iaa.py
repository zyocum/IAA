"""Loads a MAE/MAI corpus and computes Fleiss's kappa for extent tags."""

__author__ = "Zachary Yocum"
__email__  = "zyocum@brandeis.edu"

import os, numpy, fleiss
from corpus import Corpus, Document
from sys import argv, exit

def get_extent(document_id, tag):
    """Returns a string formatted extent key from a document ID and a tag."""
    start, end = get_span(tag)
    return '{}[{}:{}]'.format(document_id, start, end)

def get_span(tag):
    """Returns a (start, end) extent for the given extent tag."""
    return tag.attrib['start'], tag.attrib['end']

def index(items):
    """Create an index/codebook that maps items to integers."""
    return dict((v, k) for k, v in enumerate(sorted(items)))

def parse_name(document):
    """Parse a document's file name."""
    return os.path.basename(document.file).split('-')

def main():
    # Commandline argument parsing stuff
    default_dir='sample-data'
    directory = argv[1] if len(argv) == 2 else default_dir
    if not all([len(argv) <= 2, os.path.isdir(directory)]):
        print """Usage : python {} <directory>
    <directory> : path to directory of annotation XMLs""".format(argv[0])
        exit()
    
    corpus = Corpus(directory)
    
    # Setup accumulators
    annotator_ids = set()
    document_ids = set()
    extents = set()
    tag_types = {'NONE'}
    
    # First pass over corpus to accumulate IDs/labels
    for document in corpus:
        document_id, annotator_id, phase = parse_name(document)
        document_ids.add(document_id)
        annotator_ids.add(annotator_id)
        tag_types.update(document.extent_types)
        for tag in document.consuming_tags():
            extents.add(get_extent(document_id, tag))
    
    # Index all the things
    annotators, labels, subjects = map(
        index,
        (annotator_ids, tag_types, extents)
    )
    
    # Setup numpy array to store the data
    shape = (len(subjects), len(labels))
    data = numpy.zeros(shape, dtype=int)
    extents = dict.fromkeys(extents)
    
    # Second pass over the corpus to populate extents dictionary
    for document in corpus:
        document_id, annotator_id, phase = parse_name(document)
        for tag in document.consuming_tags():
            extent = get_extent(document_id, tag)
            if not extents[extent]:
                extents[extent] = ['NONE'] * len(annotators)
            annotator = annotators[annotator_id]
            extents[extent][annotator] = tag.tag
    
    # Final pass over the extents dictionary to populate data matrix
    for subject, annotations in extents.iteritems():
        for label in annotations:
            row, column = subjects[subject], labels[label]
            data[row,column] += 1
    
    print "Fleiss's kappa : {}".format(fleiss.kappa(data))

if __name__ == '__main__':
    main()
