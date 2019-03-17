#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import glob
import sys
import gc
import torch
from functools import partial

from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import split_corpus
import onmt.inputters as inputters
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser


def check_existing_pt_files(opt):
    pattern = opt.save_data + '.{}*.pt'
    for t in ['train', 'valid', 'vocab']:
        path = pattern.format(t)
        if glob.glob(path):
            sys.stderr.write("Please backup existing pt files: %s, "
                             "to avoid overwriting them!\n" % path)
            sys.exit(1)

def build_save_dataset(corpus_type, fields, src_lang, tgt_lang, src_reader, tgt_reader, opt):
    assert corpus_type in ['train', 'valid']
    assert src_lang in opt.langs
    assert tgt_lang in opt.langs

    direct = (src_lang, tgt_lang)

    if corpus_type == 'train':
        src = opt.train_src[direct]
        tgt = opt.train_tgt[direct]
    else:
        src = opt.valid_src[direct]
        tgt = opt.valid_tgt[direct]

    logger.info("Reading source and target files: %s %s." % (src, tgt))

    src_shards = split_corpus(src, opt.shard_size)
    tgt_shards = split_corpus(tgt, opt.shard_size)
    shard_pairs = zip(src_shards, tgt_shards)
    dataset_paths = []
    if (corpus_type == "train" or opt.filter_valid) and tgt is not None:
        filter_pred = partial(
            inputters.filter_example, use_src_len=opt.data_type == "text",
            max_src_len=opt.multi_seq_length, max_tgt_len=opt.multi_seq_length
        )
    else:
        filter_pred = None

    for i, (src_shards, tgt_shards) in enumerate(shard_pairs):
        assert len(src_shards) == len(tgt_shards)
        logger.info("Building shard %d." % i)
        dataset = inputters.MultiDataset(
            fields,
            src_lang,
            tgt_lang,
            readers=[src_reader, tgt_reader] if tgt_reader else [src_reader],
            data=([(src_lang, src_shards), (tgt_lang, tgt_shards)]
                  if tgt_reader else [(src_lang, src_shards)]),
            dirs=[None, None] if tgt_reader else [None],
            sort_key=inputters.str2sortkey[opt.data_type],
            filter_pred=filter_pred
        )

        data_path = "{:s}.{:s}.{:s}-{:s}.{:d}.pt".format(opt.data_type, src_lang, tgt_lang, corpus_type, i)
        dataset_paths.append(data_path)

        logger.info(" * saving %sth %s data shard to %s."
                    % (i, corpus_type, data_path))

        dataset.save(data_path)

        del dataset.examples
        gc.collect()
        del dataset
        gc.collect()

    return dataset_paths


def build_save_vocab(train_dataset, fields, opt):
    fields = inputters.build_vocab(
        train_dataset, fields, opt.data_type, opt.share_vocab,
        opt.src_vocab, opt.src_vocab_size, opt.src_words_min_frequency,
        opt.tgt_vocab, opt.tgt_vocab_size, opt.tgt_words_min_frequency,
        vocab_size_multiple=opt.vocab_size_multiple
    )
    vocab_path = opt.save_data + '.vocab.pt'
    torch.save(fields, vocab_path)


def count_features(path):
    with codecs.open(path, "r", "utf-8") as f:
        first_tok = f.readline().split().split(None, 1)[0]
        return len(first_tok.split(u"|")) - 1


def main(opt):
    ArgumentParser.validate_preprocess_args(opt)
    torch.manual_seed(opt.seed)
    check_existing_pt_files(opt)

    init_logger(opt.log_file)
    logger.info("Extracting features...")

    opt.langs = opt.langs.split(',')
    if len(opt.langs) > 3:
        sys.stderr.write("Only support 2 & 3 languages now!")
        sys.exit(1)

    if len(opt.langs) == 2:
        lang1 = opt.langs[0]
        lang2 = opt.langs[1]
    elif len(opt.langs) == 3:
        lang1 = opt.langs[0]
        lang2 = opt.langs[1]
        lang3 = opt.langs[2]
    else:
        sys.stderr.write("Only support 2 & 3 languages now!")

    opt.train_src = {k: v for k, v in [x.split(':') for x in opt.train_src.split(';') if len(x) > 0]}
    opt.train_src = {tuple(k.split('-')): v for k, v in opt.train_src.items()}
    opt.train_tgt = {k: v for k, v in [x.split(':') for x in opt.train_tgt.split(';') if len(x) > 0]}
    opt.train_tgt = {tuple(k.split('-')): v for k, v in opt.train_tgt.items()}

    opt.valid_src = {k: v for k, v in [x.split(':') for x in opt.valid_src.split(';') if len(x) > 0]}
    opt.valid_src = {tuple(k.split('-')): v for k, v in opt.valid_src.items()}
    opt.valid_tgt = {k: v for k, v in [x.split(':') for x in opt.valid_tgt.split(';') if len(x) > 0]}
    opt.valid_tgt = {tuple(k.split('-')): v for k, v in opt.valid_tgt.items()}

    trans_directions = [k for k, v in opt.train_src.items()]

    nfeatsdict = {}
    for lang in opt.langs:
        nfeats = 0
        for direct in trans_directions:
            if lang in direct:
                nfeats += count_features(opt.train_src[direct]) if opt.data_type == 'text' else 0
        nfeatsdict[lang] = nfeats
        logger.info(" * number of %s features: %d." % (str(lang), nfeats))

    logger.info("Building `Field` object...")

    fields = inputters.get_multi_fields(
        opt.data_type,
        opt.langs,
        trans_directions,
        nfeatsdict,
        dynamic_dict=opt.dynamic_dict,
        truncate=opt.multi_seq_length_trunc
    )

    readers = {}
    for lang in opt.langs:
        readers[lang] = inputters.str2reader["multitext"].from_opt(opt)

    train_dataset_files = {}
    for direct in trans_directions:
        logger.info("Building & saving training data...")
        (src_lang, tgt_lang) = direct
        src_reader = readers[src_lang]
        tgt_reader = readers[tgt_lang]
        train_dataset_files[direct] = build_save_dataset(
            'train', fields, src_lang, tgt_lang, src_reader, tgt_reader, opt)

    if opt.valid_src and opt.valid_tgt:
        for direct in trans_directions:
            (src_lang, tgt_lang) = direct
            logger.info("Building & saving validation data...")
            build_save_dataset('valid', fields, src_lang, tgt_lang, src_reader, tgt_reader, opt)

    logger.info("Building & saving vocabulary...")
    build_save_vocab(train_dataset_files, fields, opt)


def _get_parser():
    parser = ArgumentParser(description='preprocess.py')

    opts.config_opts(parser)
    opts.multi_preprocess_opts(parser)
    return parser


if __name__ == '__main__':
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)