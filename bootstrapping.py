#!/usr/bin/env python

import argparse
import numpy as np
import uuid
import re
import subprocess
from collections import Counter
import time
import uuid
import os

from conll_for_bootstrapping import read_conll, write_conll



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# parser.add_argument('model', help='any model, probably. I just need to know what lower_case param is')
parser.add_argument('test', help='test data in conllu format')
parser.add_argument('original_model_predictions', help='file with original model predictions')
parser.add_argument('clause_boundary_predictions', help='file with clause boundary-based predictions')
parser.add_argument('--verbose', default=False, help='print out full eval report')
parser.add_argument('--which-cuda', type=int, default=0, help='which cuda to use')
args = parser.parse_args()

print(args)

np.random.seed(22)

# print(f'loading test dataset from {args.test}')
test_data = read_conll(args.test, lower_case=True)
orig_predictions = read_conll(args.original_model_predictions, lower_case=True)
cb_predictions = read_conll(args.clause_boundary_predictions, lower_case=True)


# eval_conll_with_bootstrapping(test, orig, args.test)

sample_size = len(test_data)

orig_uas_better = 0
orig_las_better = 0

orig_uas_scores = []
orig_las_scores = []
cb_uas_scores = []
cb_las_scores = []

runs = 1000

def get_score(predictions_file, gold_file):
    p = subprocess.run(['./eval.pl', '-g', gold_file, '-s', predictions_file], stdout=subprocess.PIPE)
    o = p.stdout.decode('utf-8')
    if args.verbose: print(o)
    uas = float(re.search(r'Unlabeled attachment score:.*=\s(.+)\s%', o).group(1))
    las = float(re.search(r'Labeled   attachment score:.*=\s(.+)\s%', o).group(1))
    
    return uas, las


# do this 1,000 times
for i in range(runs):
    # start_time = time.time()
    # get a sample of ids of entries from the entry list the length of entry list
    indices_for_sample = np.random.choice(sample_size, sample_size, replace=True)
    # to use eval script, we want predictions in files
    # these will be the names for temp files
    gold_file_name = "tmp/gold" + str(uuid.uuid1()) + ".conllx"
    orig_predict_file_name = "tmp/orig_predicted" + str(uuid.uuid1()) + ".conllx"
    cb_predict_file_name = "tmp/cb_predicted" + str(uuid.uuid1()) + ".conllx"

    # get eval data based on sampled indices
    gold_sample = []
    orig_pred_sample = []
    cb_pred_sample = []

    for j, indx in enumerate(indices_for_sample):
        gold_sample.append(test_data[indx])
        orig_pred_sample.append(orig_predictions[indx])
        cb_pred_sample.append(cb_predictions[indx])

    # making sure sample size is right and that we actually resampled:
    # print("len of test set: ", sample_size)
    # print("len of sample: ", len(indices_for_sample))
    # print("len of distinct sample: ", len(list(set(indices_for_sample))))
    # # print(gold_file_name)

    # write sample data to files to be used with eval script
    write_conll(gold_file_name, gold_sample)
    write_conll(orig_predict_file_name, orig_pred_sample)
    write_conll(cb_predict_file_name, cb_pred_sample)


    # get scores
    original_uas, original_las = get_score(orig_predict_file_name, gold_file_name)
    cb_uas, cb_las = get_score(cb_predict_file_name, gold_file_name)

    print("Orig (uas and las):", original_uas, " ", original_las)
    print("New  (uas and las):", cb_uas, " ", cb_las)
    # keep track of scores to calculate mean and std
    orig_uas_scores.append(original_uas)
    orig_las_scores.append(original_las)
    cb_uas_scores.append(cb_uas)
    cb_las_scores.append(cb_las)

    if float(original_uas) > float(cb_uas):
        orig_uas_better += 1
        print("original better")

    if float(original_las) > float(cb_las):
        orig_las_better += 1
        print("original las better")

    # one_loop_time = time.time() - start_time
    # print(f"took {one_loop_time} (probably seconds)")
    
    os.remove(gold_file_name)
    os.remove(orig_predict_file_name)
    os.remove(cb_predict_file_name)

    # keep track of p value for every iteration (for debugging)

    # print("p: ", float(orig_uas_better)/(i+1))
    # print("orig better: ", orig_uas_better, "out of", i+1, "runs\n")

# calculate mean and std
orig_uas_mean = np.mean(orig_uas_scores)
orig_uas_std = np.std(orig_uas_scores)
orig_las_mean = np.mean(orig_las_scores)
orig_las_std = np.std(orig_las_scores)

cb_uas_mean = np.mean(cb_uas_scores)
cb_uas_std = np.std(cb_uas_scores)
cb_las_mean = np.mean(cb_las_scores)
cb_las_std = np.std(cb_las_scores)



textpm_string = "\\\\textsubscript{\\\\textpm".replace("\\\\", "\\")
closing_paren = "\\}"
print(f"orig uas: {round(orig_uas_mean,2)} {textpm_string} {round(orig_uas_std,2)} {closing_paren} \norig las: {round(orig_las_mean,2)} {textpm_string} {round(orig_las_std,2)}\n\n")
print(f"cb uas: {round(cb_uas_mean,2)} {textpm_string} {round(cb_uas_std,2)} {closing_paren} \ncb las: {round(cb_las_mean,2)} {textpm_string} {round(cb_las_std,2)}\n")
print("\n")
# uas p-value for all runs
print("p: ", float(orig_uas_better)/runs)
print("las p: ", float(orig_las_better)/runs)