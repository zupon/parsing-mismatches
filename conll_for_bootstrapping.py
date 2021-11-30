import re
import tempfile
import subprocess
import string
import itertools
from utils import normalize, chunker
import numpy as np
from collections import Counter
import time
import uuid
import os



def read_conll(filename, include_non_projective=True, verbose=True, lower_case=True):
    """Reads dependency annotations from CoNLL-U format"""
    return list(iter_conll(filename, include_non_projective, verbose, lower_case))



def write_conll(filename, sentences):
    """Write sentences to conllx file"""
    # print("FILENAME: ", filename)
    with open(filename, 'w') as f:
        for sentence in sentences:
            for entry in sentence:
                # print(filename, " ", entry, type(entry))
                if entry.id > 0:
                    print(str(entry), file=f)
            print('', file=f)



def eval_conll(sentences, gold_filename, verbose=True):
    with tempfile.NamedTemporaryFile(mode='w') as f:
        for sentence in sentences:
            for entry in sentence:
                if entry.id > 0:
                    print(str(entry), file=f)
            print('', file=f)
        f.flush()
        p = subprocess.run(['./eval.pl', '-g', gold_filename, '-s', f.name], stdout=subprocess.PIPE)
        o = p.stdout.decode('utf-8')
        if verbose: print(o)
        m1 = re.search(r'Unlabeled attachment score: (.+)', o)
        m2 = re.search(r'Labeled   attachment score: (.+)', o)
        return m1.group(1), m2.group(1)

def eval_conll_with_bootstrapping(sentences, orig_entries, gold_filename, verbose=True):
    all_uas_scores = []
    all_las_scores = []
    # read in gold data - passing it instead
    # clear out entries that are only punctuation based on gold data
    # create two list of entries - one gold, one predicted
    # gold_wo_punkt = []
    # predicted_wo_punkt = []
    # for sent_id, sent in enumerate(orig_entries):
    #     for entry_id, entry in enumerate(sent):
    #         word = entry.form
    #         if is_punkt(word):
    #             print("IS PUNKT " , sent_id, " ", entry_id, " ", entry)
    #             continue
    #         else:
    #             print("NOT PUNKT " , sent_id, " ", entry_id, " ", entry)
    #             gold_wo_punkt.append(orig_entries[sent_id][entry_id])
    #             predicted_wo_punkt.append(sentences[sent_id][entry_id])


            # print(sent_id, " ", entry_id, " ", entry)

    def flatten(iterator):
        for sent in iterator:
            for entry in sent:
                if entry.id > 0:
                    yield entry
    sample_size = 0
    for s in sentences:
        for e in s:
            if e.id > 0:
                sample_size += 1

    gold_entries = list(flatten(orig_entries))
    predicted_entries = list(flatten(sentences))
    print(gold_entries[1:50], "<<")
    # for i in gold_entries:
    #     print(i)
    print(sample_size, "<-")
    print(len(gold_entries) , "<<<")
    print(len(predicted_entries) , "<<<<<")
    full_start_time = time.time()


    # do this 1,000 times
    for i in range(1000):
        # start_time = time.time()
        # get a sample of ids of entries from the entry list the length of entry list
        indices_for_sample = np.random.choice(sample_size, sample_size)
        # print(Counter(indices_for_sample))
        # create the gold file
        print("len indices to sample ", len(indices_for_sample))


        counter = 0

        predict_file_name = "tmp/predicted" + str(uuid.uuid1()) + ".txt"
        gold_file_name = "tmp/gold" + str(uuid.uuid1()) + ".txt"
        with open(predict_file_name, mode='w') as f:
            with open(gold_file_name, mode="w") as gf:
            # here, get rid of entries that are punctuation, sample from those, and write to file
            # to a different file, write the gold prediction file with same ids---need to read in gold file
                for ind in indices_for_sample:

                    print(str(predicted_entries[ind]), file=f)
                    print(str(gold_entries[ind]), file=gf)
                    # if predicted_entries[ind].form != gold_entries[ind].form:
                    #     print(">>", gold_entries[ind])
                    #     print(">>>", predicted_entries[ind])
                f.flush()
                gf.flush()
            # for sentence in sentences:
            #     for entry in sentence:
            #         if entry.id > 0:
            #             print(str(entry), file=f)
            #     print('', file=f)
            # f.flush()
            p = subprocess.run(['./eval.pl', '-q', '-g', gf.name, '-s', f.name], stdout=subprocess.PIPE)
            o = p.stdout.decode('utf-8')
            if verbose: print(o)
            # float these and add them to the list of all scores
            m1 = re.search(r'Unlabeled attachment score:.*=\s(.+)\s%', o).group(1)
            m2 = re.search(r'Labeled   attachment score:.*=\s(.+)\s%', o).group(1)
            # print("M1: ", m1)
            # print("M2: ", m2)
            all_uas_scores.append(float(m1))
            all_las_scores.append(float(m2))

            os.remove(predict_file_name)
            os.remove(gold_file_name)
        # one_loop_time = time.time() - start_time
        # print(f"took {one_loop_time} (probably seconds)")
    full_time = time.time() - full_start_time
    print(f"took {full_time} (probably seconds)")


    print("MADE IT HERE")
    # print(all_uas_scores, " ", type(all_uas_scores))
    # print(all_uas_scores, " ", type(all_las_scores))
    uas_mean = np.mean(all_uas_scores)
    uas_std = np.std(all_uas_scores)
    las_mean = np.mean(all_las_scores)
    las_std = np.std(all_las_scores)
    textpm_string = "\\\\textpm".replace("\\\\", "\\")
    print("latex printout: ", f" uas: {round(uas_mean,2)} {textpm_string} {round(uas_std,2)} las: {round(las_mean,2)} {textpm_string} {round(las_std,2)}")

    # print(f"uas: ${uas_mean} +- ${uas_std} las: ${las_mean} +- ${las_std}")



    return m1, m2

def parse_conll(parser, sentences, batch_size, clear=True):
    if clear:
        clear_dependencies(sentences)
    for batch in chunker(sentences, batch_size):
        parser.parse_conll(batch)
    if parser.mode == 'evaluation' and parser.print_nr_of_cycles:
        print("Nr of cycles: ", parser.nr_of_cycles, ", ", len(sentences), " ", parser.nr_of_cycles / len(sentences))



def clear_dependencies(sentences):
    for sentence in sentences:
        for entry in sentence:
            entry.head = None
            entry.deprel = None
            entry.pos = None



def iter_conll(filename, include_non_projective=True, verbose=True, lower_case=True):
    """Reads dependency annotations in CoNLL-U format and returns a generator."""
    read = 0
    non_proj = 0
    dropped = 0
    root = ConllEntry(id=0, form='<root>', upos='<root>', xpos='<root>', head=0, deprel='rroot', boundary="root")
    with open(filename) as f:
        sentence = [root]
        for line in f:
            # print("Line: ", line)
            if line.isspace() and len(sentence) > 1:
                if is_projective(sentence):
                    yield sentence
                else:
                    non_proj += 1
                    if include_non_projective:
                        yield sentence
                    else:
                        dropped += 1
                read += 1
                sentence = [root]
                continue
            entry = ConllEntry.from_line(line, lower_case=lower_case)
            sentence.append(entry)
        # we may still have one sentence in memory
        # if the file doesn't end in an empty line
        if len(sentence) > 1:
            if is_projective(sentence):
                yield sentence
            else:
                non_proj += 1
                if include_non_projective:
                    yield sentence
                else:
                    dropped += 1
            read += 1
    if verbose:
        print(f'{read:,} sentences read.')
        print(f'{non_proj:,} non-projective sentences found, {dropped:,} dropped.')
        print(f'{read-dropped:,} sentences remaining.')



def is_projective(sentence):
    """returns true if the sentence is projective"""
    roots = list(sentence)
    # keep track of number of children that haven't been
    # assigned to each entry yet
    unassigned = {
        entry.id: sum(1 for e in sentence if e.head == entry.id)
        for entry in sentence
    }
    # we need to find the parent of each word in the sentence
    for _ in range(len(sentence)):
        # only consider the forest roots
        for i in range(len(roots) - 1):
            # attach entries if:
            #   - they are parent-child
            #   - they are next to each other
            #   - the child has already been assigned all its children
            if roots[i].head == roots[i+1].id and unassigned[roots[i].id] == 0:
                unassigned[roots[i+1].id] -= 1
                del roots[i]
                break
            if roots[i+1].head == roots[i].id and unassigned[roots[i+1].id] == 0:
                unassigned[roots[i].id] -= 1
                del roots[i+1]
                break
    # if more than one root remains then it is not projective
    return len(roots) == 1



class ConllEntry:

    def __init__(self, id=None, form=None, lemma=None, upos=None, xpos=None,
                 feats=None, head=None, deprel=None, deps=None, misc=None, boundary=None, lower_case=True):
        # conll-u fields
        self.id = id
        self.form = form
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.feats = feats
        self.head = head
        self.deprel = deprel
        self.deps = deps
        self.misc = misc
        self.boundary = boundary
        # normalized token
        self.norm = normalize(form, to_lower=lower_case)
        # relative position of token's head
        self.pos = 0 if self.head == 0 else self.head - self.id

    def __repr__(self):
        return f'<ConllEntry: {self.form}>'

    def __str__(self):
        fields = [
            self.id,
            self.form,
            self.lemma,
            self.upos,
            self.xpos,
            self.feats,
            self.head,
            self.deprel,
            self.deps,
            self.misc,
            self.boundary
        ]
        return '\t'.join('_' if f is None else str(f) for f in fields)

    def get_partofspeech_tag(self, pos_type):
        if pos_type == 'upos':
            return self.upos
        elif pos_type == 'xpos':
            return self.xpos
        else:
            raise ValueError(f"Unknown type; {pos_type}")

    @staticmethod
    def from_line(line, lower_case=True):
        split_line = line.strip().split('\t')
        # print("len split line: ", len(split_line))
        if len(split_line) == 9:
            fields = [None if f == '_' else f for f in split_line]#[:-1]
        else:
            fields = [None if f == '_' else f for f in split_line][:-1]
        if fields[1] is None:
            fields[1] = '_'
        fields[0] = int(fields[0]) # id
        fields[6] = int(fields[6]) # head
        fields[7] = str(fields[7]) # deps
        return ConllEntry(*fields, lower_case=lower_case)
