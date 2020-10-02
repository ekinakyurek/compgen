#!/usr/bin/env python3

from collections import Counter, defaultdict
import json
import numpy as np
import os
from vocab import Vocab
import textdistance
import sys

CONLL_ROOT  = sys.argv[1]
CONLL_PATH  = CONLL_ROOT + "/task1/all"
TARGET_LANG = sys.argv[2]
OUTPUT_PATH = CONLL_ROOT + "/" + TARGET_LANG
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH )

TRAIN_SIZE = "medium"
TRAIN_COUNT = {"low": 100, "medium": 1000, "high": 10000}[TRAIN_SIZE]
TEST_COUNT = 100

OTHER_LANGS = []
NEIGHBORS = 4

VAL_TAG = "PST"
TEST_TAG = "FUT"

VAL_FRAC = 0.2
MAX_DIST = 10

TEST_TAG_TOKENS = 20

vocab = Vocab()
SEP = "<sep>"
vocab.add(SEP)

#HINTS = 40

rand = np.random.RandomState(0)
test_rand = np.random.RandomState(1)

def load_file(key, prefix_tokens=()):
    out_seqs = []
    out_keys = {}
    with open(os.path.join(CONLL_PATH, key)) as reader:
        for line in reader:
            lemma, inflected, morph = line.strip().split("\t")
            tokens = tuple(inflected)
            lem_tokens = tuple(lemma)
            for token in prefix_tokens:
                vocab.add(token)
            for token in tokens:
                vocab.add(token)
            for token in lem_tokens:
                vocab.add(token)
            tags = tuple(morph.split(";"))
            for tag in tags:
                vocab.add(tag)

            seq = prefix_tokens + lem_tokens + tags + (SEP,) + tokens
            enc_seq = tuple(vocab.encode(seq))
            out_seqs.append(enc_seq)
            out_keys[enc_seq] = (prefix_tokens, tags, lemma)

    return out_seqs, out_keys

def permutation(rand, seqs):
    seqs = rand.permutation(seqs)
    seqs = [tuple(int(i) for i in seq) for seq in seqs]
    return seqs

def build_stratified(hints):
    seqs = []
    neighborhood_keys = {}
    for lang in [TARGET_LANG] + OTHER_LANGS:
        train_key = f"{lang}-train-high"
        prefix = (f"<{lang}>",)
        lang_seqs, lang_neighborhood_keys = load_file(train_key, prefix_tokens=prefix)
        seqs += lang_seqs
        neighborhood_keys.update(lang_neighborhood_keys)
    seqs = sorted(seqs)

    def project_tag(tag):
        return tag[:2]
    tag_types = sorted(set(project_tag(k) for k in neighborhood_keys.values()))

    val_types = [t for t in tag_types if f"<{TARGET_LANG}>" in t[0] and VAL_TAG in t[1]]
    test_types = [t for t in tag_types if f"<{TARGET_LANG}>" in t[0] and TEST_TAG in t[1]]
    train_types = [t for t in tag_types if not (t in val_types or t in test_types)]

    train_like_seqs = [seq for seq in seqs if project_tag(neighborhood_keys[seq]) in train_types]
    # sorry
    val_like_seqs = [
        permutation(test_rand, [seq for seq in seqs if project_tag(neighborhood_keys[seq]) == tag])[:TEST_TAG_TOKENS]
        for tag in val_types
    ]
    #val_like_seqs = [item for seq in val_like_seqs for item in seq]
    #val_like_seqs = [seq for seq in seqs if project_tag(neighborhood_keys[seq]) in val_types]
    test_like_seqs = [
        permutation(test_rand, [seq for seq in seqs if project_tag(neighborhood_keys[seq]) == tag])[:TEST_TAG_TOKENS]
        for tag in test_types
    ]
    train_like_seqs = permutation(rand, train_like_seqs)
    #test_like_seqs = [item for seq in test_like_seqs for item in seq]
    #test_like_seqs = [seq for seq in seqs if project_tag(neighborhood_keys[seq]) in test_types]

    #val_like_train_seqs = [ss for s in val_like_seqs for ss in s[:2]]
    #val_like_val_seqs = [ss for s in val_like_seqs for ss in s[2:]]
    #np.random.shuffle(val_like_train_seqs)
    #np.random.shuffle(val_like_val_seqs)
    #test_like_train_seqs = [ss for s in test_like_seqs for ss in s[:2]]
    #test_like_test_seqs = [ss for s in test_like_seqs for ss in s[2:]]
    #np.random.shuffle(test_like_train_seqs)
    #np.random.shuffle(test_like_test_seqs)
    val_like_seqs = [ss for s in val_like_seqs for ss in s]
    val_like_seqs = permutation(test_rand, val_like_seqs)
    val_like_train_seqs = val_like_seqs[:hints]
    for seq in val_like_train_seqs:
        print("tv", " ".join(vocab.decode(seq)))
    val_like_val_seqs = val_like_seqs[hints:]
    test_like_seqs = [ss for s in test_like_seqs for ss in s]
    test_like_seqs = permutation(test_rand, test_like_seqs)
    test_like_train_seqs = test_like_seqs[:hints]
    test_like_test_seqs = test_like_seqs[hints:]

    train_easy_seqs = train_like_seqs[:TRAIN_COUNT-2*hints]
    train_hard_seqs = val_like_train_seqs + test_like_train_seqs
    val_easy_seqs = train_like_seqs[TRAIN_COUNT:TRAIN_COUNT+TEST_COUNT]
    val_hard_seqs = val_like_val_seqs[:TEST_COUNT]
    assert len(val_hard_seqs) == TEST_COUNT
    test_easy_seqs = train_like_seqs[TRAIN_COUNT+TEST_COUNT:TRAIN_COUNT+2*TEST_COUNT]
    test_hard_seqs = test_like_test_seqs[:TEST_COUNT]
    assert len(test_hard_seqs) == TEST_COUNT
    print("train",
        len(train_easy_seqs),
        len(train_hard_seqs),
        len([t for t in val_like_train_seqs if project_tag(neighborhood_keys[t]) in val_types]),
        len([t for t in test_like_train_seqs if project_tag(neighborhood_keys[t]) in test_types]),
    )
    print("val", len(val_easy_seqs), len(val_hard_seqs))
    print("test", len(test_easy_seqs), len(test_hard_seqs))
    return (
        train_easy_seqs,
        train_hard_seqs,
        val_easy_seqs,
        val_hard_seqs,
        test_easy_seqs,
        test_hard_seqs,
        neighborhood_keys
    )

#def build_standard():
#    train_seqs = []
#    val_seqs = []
#    test_seqs = []
#    neighborhood_keys = {}
#
#    for lang in LANGS:
#        train_key = f"{lang}-train-{TRAIN_SIZE}"
#        test_key = f"{lang}-test"
#
#        prefix = (f"<{lang}>",)
#
#        train, train_neighborhood_keys = load_file(train_key, prefix_tokens=prefix)
#        n_val = int(len(train) * VAL_FRAC)
#        val = train[-n_val:]
#        train = train[:-n_val]
#
#        test, test_neighborhood_keys = load_file(test_key, prefix_tokens=prefix)
#
#        train_seqs += train
#        val_seqs += val
#        test_seqs += test
#        neighborhood_keys.update(train_neighborhood_keys)
#        neighborhood_keys.update(test_neighborhood_keys)
#
#    return train_seqs, val_seqs, test_seqs, neighborhood_keys

def build_dataset(hints):
    (
        train_easy_seqs,
        train_hard_seqs,
        val_easy_seqs,
        val_hard_seqs,
        test_easy_seqs,
        test_hard_seqs,
        neighborhood_keys,
    ) = build_stratified(hints)

    train_seqs = train_easy_seqs + train_hard_seqs
    seqs = train_seqs + val_easy_seqs + val_hard_seqs + test_easy_seqs + test_hard_seqs
    to_train = len(train_seqs)
    to_val_easy = to_train + len(val_easy_seqs)
    to_val_hard = to_val_easy + len(val_hard_seqs)
    to_test_easy = to_val_hard + len(test_easy_seqs)
    to_test_hard = to_test_easy + len(test_hard_seqs)
    splits = {
        "train": list(range(0, to_train)),
        "val_easy": list(range(to_train, to_val_easy)),
        "val_hard": list(range(to_val_easy, to_val_hard)),
        "test_easy": list(range(to_val_hard, to_test_easy)),
        "test_hard": list(range(to_test_easy, to_test_hard)),
    }

    neighborhoods = {}
    for i, seq in enumerate(seqs):
        if i % 100 == 0:
            print(f"{i} / {len(seqs)}")
        lang, morph, lem = neighborhood_keys[seq]
        morph = set(morph)

        neighbors1 = []
        for i1, seq1 in enumerate(train_seqs):
            if i1 == i:
                continue
            lang1, morph1, lem1 = neighborhood_keys[seq1]
            morph1 = set(morph1)
            if lang1 != lang:
                continue
            if i % 2 == 0 and morph1 == morph:
                continue
            score = (len(morph ^ morph1), textdistance.jaccard(lem, lem1))
            neighbors1.append((score, i1, morph1))
        rand.shuffle(neighbors1)

        out = []
        for _, i1, morph1 in sorted(neighbors1)[:NEIGHBORS]:

            neighbors2 = []
            for i2, seq2 in enumerate(train_seqs):
                if i2 == i or i2 == i1:
                    continue
                lang2, morph2, lem2 = neighborhood_keys[seq2]
                morph2 = set(morph2)
                if lang2 != lang:
                    continue
                if morph2 == morph or morph2 == morph1:
                    continue
                if len(morph - morph1 - morph2) > 0:
                    continue
                score = len(morph1 ^ morph2)
                neighbors2.append((score, i2, morph2))
            #neighbors2 = [n for n in neighbors2 if n[0] <= 2]

            if len(neighbors2) > 0:
                _, i2, morph2 = min(neighbors2)
                out.append([i1, i2])
            #elif i in splits["train"]:
            else:
                # fake references only for training data
                # TODO just don't train on these?
                _, i2, morph2 = min(n for n in neighbors1 if n[1] != i1)
                out.append(rand.permutation([i1, i2]).tolist())
                #print("warning", morph, morph1)
            #else:
            #    assert False, (morph, morph1)

        if len(out) == 0:
            assert False
            #if i in splits["train"]:
            #    assert False
            #else:
            #    split, = (k for k in splits if i in splits[k])
            #    splits[split].remove(i)
        else:
            if i in splits["train"][-10-len(train_hard_seqs):]:
                print(" ".join(vocab.decode(seqs[i])))
                print(" ".join(vocab.decode(seqs[i1])))
                for score, i2, morph2 in sorted(neighbors2[:NEIGHBORS]):
                    print(" ", score, " ".join(vocab.decode(seqs[i2])))
                print()
            neighborhoods[i] = out
    return seqs, neighborhoods, splits

def main():
    for hints in (4, 8, 16):
        for i in range(5):
            seqs, neighborhoods, splits = build_dataset(hints)
            with open(f"{OUTPUT_PATH}/seqs.hints-{hints}.{i}.json", "w") as writer:
                json.dump(seqs, writer)
            with open(f"{OUTPUT_PATH}/neighborhoods.hints-{hints}.{i}.json", "w") as writer:
                json.dump(neighborhoods, writer)
            with open(f"{OUTPUT_PATH}/splits.hints-{hints}.{i}.json", "w") as writer:
                json.dump(splits, writer)

    with open(f"{OUTPUT_PATH}/vocab.json", "w") as writer:
        vocab.dump(writer)

if __name__ == "__main__":
    main()
