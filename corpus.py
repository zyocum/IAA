"""Module for working with MAE/MAI annotation documents."""

__author__ = "Zachary Yocum"
__email__  = "zyocum@brandeis.edu"

import os, re
from warnings import warn
from xml.etree import ElementTree

class Corpus(object):
    """A class for working with collections of Documents."""
    def __init__(self, directory, pattern='.*\.xml', recursive=True):
        super(Corpus, self).__init__()
        self.directory = directory
        self.pattern = pattern
        self.recursive = recursive
        self.validate()
    
    def __repr__(self):
        return '<{name} with {n} documents>'.format(
            name=self.__class__.__name__,
            n=len(self.documents())
        )
    
    def __iter__(self):
        return iter(self.documents())
    
    def documents(self):
        files = find_files(self.directory, self.pattern, self.recursive)
        return map(Document, files)
    
    def validate(self):
        return all(map(Document.validate, self.documents()))
        
class Document(object):
    """A MAE/MAI annotation document."""
    def __init__(self, file):
        self.file = file
        self.tree = ElementTree.parse(file)
        self.root = self.tree.getroot()
        self.task = self.root.tag
        self.text = self.root.find('TEXT').text
        self.tags = self.root.find('TAGS').getchildren()
        self.tag_types = set(tag.tag for tag in self.tags)
        self.extent_types = set(tag.tag for tag in self.extent_tags())
        self.link_types = set(tag.tag for tag in self.link_tags())
    
    def __repr__(self):
        return "Document({})".format(file.name)
    
    def __str__(self):
        return self.text.encode('utf-8')
    
    def extent_tags(self):
        return (tag for tag in self.tags if 'start' in tag.attrib)
    
    def link_tags(self):
        return (tag for tag in self.tags if 'from' in tag.attrib)
    
    def consuming_tags(self):
        tags = self.extent_tags()
        return (tag for tag in tags if int(tag.attrib['start']) > -1)
    
    def non_consuming_tags(self):
        tags = self.extent_tags()
        return (tag for tag in tags if int(tag.attrib['start']) <= -1)
    
    def validate(self):
        is_valid = True
        if not self.tags:
            is_valid = False
            warning = "No tag elements found\n\tFile : '{}'".format(self.file)
            warn(warning, RuntimeWarning)
        for tag in self.consuming_tags():
            start, end = map(int, (tag.attrib['start'], tag.attrib['end']))
            text_attribute = tag.attrib['text']
            text_span = self.text[slice(start, end)].replace('\n', ' ')
            if text_attribute != text_span:
                is_valid = False
                warning = '\n\t'.join([
                    'Misaligned extent tag',
                    "File   : '{file}'",
                    "Extent : [{start}:{end}]",
                    "Tag ID : '{id}'",
                    "Text   : '{text_attribute}'",
                    "Span   : '{text_span}'"
                ]).format(
                    file=self.file,
                    start=start,
                    end=end,
                    id=tag.attrib['id'],
                    text_attribute=text_attribute,
                    text_span=text_span
                ).encode('utf-8')
                warn(warning, RuntimeWarning)
        return is_valid
    
def find_files(directory='.', pattern='.*', recursive=True):
    if recursive:
        return (os.path.join(directory, filename)
            for directory, subdirectories, filenames in os.walk(directory)
            for filename in filenames if re.match(pattern, filename))
    else:
        return (os.path.join(directory, filename)
            for filename in os.listdir(directory)
            if re.match(pattern, filename))
