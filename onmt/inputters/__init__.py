"""Module defining inputters.

Inputters implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of embeddings.
"""
from onmt.inputters.inputter import \
    load_old_vocab, get_fields, get_multi_fields, OrderedIterator, \
    build_vocab, old_style_vocab, filter_example
from onmt.inputters.dataset_base import Dataset, MultiDataset
from onmt.inputters.text_dataset import text_sort_key, TextDataReader, MultiTextDataReader
from onmt.inputters.image_dataset import img_sort_key, ImageDataReader
from onmt.inputters.audio_dataset import audio_sort_key, AudioDataReader
from onmt.inputters.datareader_base import DataReaderBase


str2reader = {
    "text": TextDataReader, "multitext": MultiTextDataReader, "img": ImageDataReader, "audio": AudioDataReader}
str2sortkey = {
    'text': text_sort_key, 'img': img_sort_key, 'audio': audio_sort_key}


__all__ = ['Dataset', 'MultiDataset', 'load_old_vocab',
           'get_fields', 'get_multi_fields',
           'DataReaderBase',
           'filter_example', 'old_style_vocab',
           'build_vocab', 'OrderedIterator',
           'text_sort_key', 'img_sort_key', 'audio_sort_key',
           'TextDataReader', 'MultiTextDataReader', 'ImageDataReader', 'AudioDataReader']
