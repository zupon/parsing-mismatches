import sys
from tqdm import tqdm
from collections import Counter
import functions as f

# set hyperparameters
noise_type = ["none", "freq", "10_types", "20_types", "30_types", "40_types", "50_types", "10_tokens", "20_tokens", "30_tokens", "40_tokens", "50_tokens"]
# noise_type = ["10_types", "20_types", "30_types", "40_types", "50_types"]
# noise_type = ["none", "freq"]
top_n = 20
threshold = 0.0

# load vectors
print("loading vectors...")
vectors_gd = "bilingual_vectors_scannell.magnitude"
vectors_ga = "bilingual_vectors_scannell.magnitude"

for noise in noise_type:
    print(f"\nGenerating training data based on noise_type: {noise}\n")

    corpus_gd = "train_data/gd/gd_noise="+str(noise)+"-ud-train.conllu"
    corpus_ga = "train_data/ga/ga_noise=none-ud-train.conllu"

    # train data
    list_gd, sentences_gd = f.process_training_data(corpus_gd)
    list_ga, sentences_ga = f.process_training_data(corpus_ga)

    # vector conversion only
    conversion_gd = f.get_conversions_bilingual_vectors(sentences_gd,
                                                    list_gd, list_ga,
                                                    vectors_gd, vectors_ga)
    converted_corpus_gd = corpus_gd[:-16]+"_conversion=bilingual_vectors-ud-train.conllu"

    i = 0
    j = len(conversion_gd)
    for item in conversion_gd:
        if len(list(conversion_gd[item])) != 0:
    #         print(item, conversion_gd[item])
            i += 1

    print(f"Total word pairs:\t{j}")
    print(f"Word pairs w/ other relations in Irish (train):\t{i}")

    (total_lines, changed) = f.apply_conversions_bilingual(corpus_gd, converted_corpus_gd, conversion_gd)
    print(f"Total lines:\t{total_lines}")
    print(f"Lines changed:\t{changed}")
    
    
    print(f"\nGenerating dev data based on noise_type: {noise}\n")
    corpus_gd = "dev_data/gd/gd_noise="+str(noise)+"-ud-dev.conllu"
    corpus_ga = "dev_data/ga/ga_noise=none-ud-dev.conllu"

    # dev data
    list_gd, sentences_gd = f.process_training_data(corpus_gd)
    list_ga, sentences_ga = f.process_training_data(corpus_ga)

    # vector conversion only
    conversion_gd = f.get_conversions_bilingual_vectors(sentences_gd,
                                                    list_gd, list_ga,
                                                    vectors_gd, vectors_ga)
    converted_corpus_gd = corpus_gd[:-14]+"_conversion=bilingual_vectors-ud-dev.conllu"

    i = 0
    j = len(conversion_gd)
    for item in conversion_gd:
        if len(list(conversion_gd[item])) != 0:
    #         print(item, conversion_gd[item])
            i += 1

    print(f"Total word pairs:\t{j}")
    print(f"Word pairs w/ other relations in Irish (dev):\t{i}")

    (total_lines, changed) = f.apply_conversions_bilingual(corpus_gd, converted_corpus_gd, conversion_gd)
    print(f"Total lines:\t{total_lines}")
    print(f"Lines changed:\t{changed}")
