# helper functions for getting corpus comparison
from collections import Counter
from tqdm import tqdm
from pymagnitude import *


def load_corpus(language):
    """
    Loads corpus files depending on selected language.
    """
    print(f"Loading {language} corpora...")
    if language=="english":
        # GUM (Georgetown University Multilayer) corpus from UD website
        corpus_A_dir = "data/ud-en-gum/"
        train_A = corpus_A_dir+"en_gum-ud-train.conllu"
        dev_A = corpus_A_dir+"en_gum-ud-dev.conllu"

        # wsj corpus with different conventions
        # converted from Stanford dependencies (?)
        corpus_B_dir = "data/wsj-DIFF-CONVENTIONS/"
        train_B = corpus_B_dir+"train.conllu"
        dev_B = corpus_B_dir+"dev.conllu"
        test_B = corpus_B_dir+"test.conllu"
    elif language == "persian":
        # original seraji
        corpus_A_dir = "data/original_seraji/"
        train_A = corpus_A_dir+"fa_seraji-ud-train-removeLines.conllu"
        dev_A = corpus_A_dir+"fa_seraji-ud-test.conllu"
        # new seraji
#         corpus_A_dir = "data/new_seraji/"
#         train_A = corpus_A_dir+"fa_seraji-ud-train.conllu"
#         dev_A = corpus_A_dir+"fa_seraji-ud-train.conllu"
        # PerDT corpus
        corpus_B_dir = "data/PerDT/"
        train_B = corpus_B_dir+"fa_perdt-ud-train-removeLines.conllu"
        dev_B = corpus_B_dir+"fa_perdt-ud-test.conllu"
        
    else:
        print("Please select a language!")
    print("Corpora loaded!")

    return train_A, train_B


def load_vectors(language):
    """
    Loads vector file depending on selected language.
    """
    if language == "english":
        vector_file = "./glove.840B.300d.magnitude"
#         vector_file = "./en-dep-based-vecs-words.magnitude"
    elif language == "persian":
        vector_file = "./cc.fa.300.magnitude"
    else:
        print("Please select a language!")
    if len(vector_file) > 0:    
        return vector_file
    else:
        print(f"No vector file found for {language}!")

        
def process_training_data(filename):
    with open(filename) as infile:
        file_text = infile.read()
        
    sents = file_text.rstrip().split("\n\n")
    num_tokens = 0
    num_sents = len(sents)
    
    pair_to_relations = {}
    pair_relation_to_sentences = {}
    
    for sent in sents:
        sentence_text = []
        for line in sent.split("\n"):
            word = line.split("\t")[1]
            sentence_text.append(word)
        for line in sent.split("\n"):
            line = line.split("\t")
            word_idx = line[0]
            word = line[1]
            num_tokens += 1
            head_idx = int(line[6])
            head_word = sent.split("\n")[head_idx-1].split("\t")[1] if head_idx != 0 else "#ROOT#"
            relation = line[7]
            word_pair = (head_word, word)
            if word_pair in pair_to_relations:
                pair_to_relations[word_pair].append(relation)
            else:
                pair_to_relations[word_pair] = [relation]
            pair_relation = (word_pair, relation)
            if pair_relation in pair_relation_to_sentences:
                pair_relation_to_sentences[pair_relation].append(" ".join(sentence_text))
            else:
                pair_relation_to_sentences[pair_relation] = [" ".join(sentence_text)]
        
    print(f"{num_sents:,} sentences processed")
    print(f"{num_tokens:,} tokens processed")
    print(f"Average sentence length:\t{num_tokens/num_sents}")
    print(f"{len(pair_to_relations):,} head-dependent:relation pairs")
    print(f"{len(pair_relation_to_sentences):,} head-dependent-relation:sentence triples")
    
    return pair_to_relations, pair_relation_to_sentences

    
# def process_ud_data(ud_file, num_sents):
#     """
#     Processes UD data into dictionaries of word pairs, relations, and sentences.
#     """
#     print(f"\nProcessing '{ud_file}' data file...")
#     with open(ud_file) as infile:
#         ud_lines = infile.readlines()
#     num_tokens = 0
#     # get list of sentences
#     sentences = []
#     sentence = []
#     for line in ud_lines:
# #         print(line)
#         if "# sent_id = " in line:
#             sentence.append(line.split(" sent_id = ")[1].strip())
#         if "# text =" in line:
#             sentence.append(line.split(" text =")[1].strip())
#         elif line[0] == "#":
#             continue
#         elif len(line.strip()) == 0:
#             sentences.append(sentence)
#             sentence = []
#         else:
#             split = line.split("\t")
#             sentence.append(split)

#     pair_to_relations = {}
#     pair_relation_to_sentences = {}
#     num_sents = min(len(sentences), num_sents)
#     for sentence in tqdm(sentences[:num_sents]):
# #         print("\nSENTENCE:\t",sentence)
#         # each sentence is a list of lines split on tabs
#         sentence_id = sentence[0]
#         sentence_text = sentence[1]
# #         print("\nSentence ID:\t"+sentence_id)
# #         print("Sentence Text:\t"+sentence_text)
#         for line in sentence[2:]:
# #             print("LINE:\t",line)
#             word_idx = line[0]
# #             print("word idx:\t",word_idx)
#             word = line[1]
#             num_tokens += 1
# #             print("word:\t"+word)
#             head_idx = int(line[6])
# #             print("head idx:\t",head_idx)
# #             head_idx+1 needed to match place in list
#             head_word = sentence[head_idx+1][1] if head_idx != 0 else "#ROOT#"
# #             print("head word:\t"+head_word)
#             relation = line[7]
# #             if relation not in gum_relations:
# #                 gum_relations.append(relation)
# #             print("relation:\t"+relation)
#             word_pair = (head_word, word)
# #             print("word pair:\t",word_pair)
#             if word_pair in pair_to_relations:
#                 pair_to_relations[word_pair].append(relation)
#             else:
#                 pair_to_relations[word_pair] = [relation]
#             pair_relation = (word_pair, relation)
#             if pair_relation in pair_relation_to_sentences:
#                 pair_relation_to_sentences[pair_relation].append([sentence_id, sentence_text])
#             else:
#                 pair_relation_to_sentences[pair_relation] = [[sentence_id, sentence_text]]
#     """
#     pair_to_relations dict:
#         e.g. {("Need", "'ll"): ["aux"], ...}

#     pair_relation_to_sentences dict:
#         e.g. {(("funnel", "Large"), "amod"): ["Large funnel or strainer to hold filter"], ...}
#     """
#     print(f"{num_sents:,} sentences processed")
#     print(f"{num_tokens:,} tokens processed")
#     print(f"Average sentence length:\t{num_tokens/num_sents}")
#     print(f"{len(pair_to_relations):,} head-dependent:relation pairs")
#     print(f"{len(pair_relation_to_sentences):,} head-dependent-relation:sentence triples")
          
#     return pair_to_relations, pair_relation_to_sentences
     


# def process_wsj_data(ud_file, num_sents):
#     """
#     Processes WSJ data into dictionaries of word pairs, relations, and sentences.
#     """
#     print(f"\nProcessing '{ud_file}' data file...")
#     with open(ud_file) as infile:
#         ud_lines = infile.readlines()
#     num_tokens = 0
#     # get list of sentences
#     sentences = []
#     sentence = []
#     for line in ud_lines:
#         if "# sent_id = " in line:
#             sentence.append(line.split(" sent_id = ")[1].strip())
#         if "# text =" in line:
#             sentence.append(line.split(" text = ")[1].strip())
#         elif line[0] == "#":
#             continue
#         elif len(line.strip()) == 0:
#             sentences.append(sentence)
#             sentence = []
#         else:
#             split = line.split("\t")
#             sentence.append(split)

#     pair_to_relations = {}
#     pair_relation_to_sentences = {}
#     num_sents = min(len(sentences), num_sents)
#     for sentence in tqdm(sentences[:num_sents]):
# #         print("\nSENTENCE:\t",sentence)
#         # each sentence is a list of lines split on tabs
#         sentence_id = sentences.index(sentence)
# #         print("Sentence ID:\t"+str(sentence_id))
#         sentence_text = []
#         for line in sentence:
#             word = line[1]
#             sentence_text.append(word)
# #         print("Sentence Text:\t"+" ".join(sentence_text))
#         for line in sentence:
# #             print("LINE:\t",line)
#             word_idx = line[0]
# #             print("word idx:\t",word_idx)
#             word = line[1]
#             num_tokens += 1
# #             print("word:\t"+word)
#             head_idx = int(line[6])
# #             print("head idx:\t",head_idx)
# #             head_idx+1 needed to match place in list
#             head_word = sentence[head_idx-1][1] if head_idx != 0 else "#ROOT#"
# #             print("head word:\t"+head_word)
#             relation = line[7]
# #             if relation not in wsj_relations:
# #                 wsj_relations.append(relation)
# #             print("relation:\t"+relation)
#             word_pair = (head_word, word)
# #             print("word pair:\t",word_pair)
#             if word_pair in pair_to_relations:
#                 pair_to_relations[word_pair].append(relation)
#             else:
#                 pair_to_relations[word_pair] = [relation]
#             pair_relation = (word_pair, relation)
#             if pair_relation in pair_relation_to_sentences:
#                 pair_relation_to_sentences[pair_relation].append([sentence_id, " ".join(sentence_text)])
#             else:
#                 pair_relation_to_sentences[pair_relation] = [[sentence_id, " ".join(sentence_text)]]
#     """
#     pair_to_relations dict:
#         e.g. {("Need", "'ll"): ["aux"], ...}

#     pair_relation_to_sentences dict:
#         e.g. {(("funnel", "Large"), "amod"): ["Large funnel or strainer to hold filter"], ...}
#     """
#     print(f"{num_sents:,} sentences processed")
#     print(f"{num_tokens:,} tokens processed")
#     print(f"Average sentence length:\t{num_tokens/num_sents}")
#     print(f"{len(pair_to_relations):,} head-dependent:relation pairs")
#     print(f"{len(pair_relation_to_sentences):,} head-dependent-relation:sentence triples")
#     return pair_to_relations, pair_relation_to_sentences


def generate_human_readable_output(filename, mismatches, sentences, this_data, other_data):
    """
    Generates human-readable output file for evaluation.
    
    Args:
        filename: the output filename
        
        mismatches: the dictionary of word pairs with a relation in one file but not the other
        
        sentences: the dictionary of word pairs, their relations, and the sentences they're found in
    """
    conversion_dict = {}
    with open(filename, "w") as output:
        header = ("SENTENCE" + "\t" + 
                  "HEAD WORD" + "\t" + 
                  "DEPENDENT WORD" + "\t" + 
                  "RELATION" + "\t" + 
                  "TOP RELATION IN THIS DATA" + "\t" +
                  "COUNT OF TOP RELATION IN THIS DATA" + "\t" +
                  "PROPORTION OF TOP RELATION IN THIS DATA" + "\t"
                  "TOP RELATION IN OTHER DATA" + "\t" +
                  "COUNT OF TOP RELATION IN OTHER DATA" + "\t" +
                  "PROPORTION OF TOP RELATION IN OTHER DATA" + "\n")
        output.write(header)
        for pair in mismatches:
#             print("word pair:\t", pair)
            head_word = pair[0]
            dependent_word = pair[1]
            # relations for this pair in this partition / corpus
            these_relations = Counter(this_data[pair])
            # relations for this pair in other partition / corpus
            other_relations = Counter(other_data[pair])
#             print("other_relations:\t", sum(Counter(other_relations).values()))
            for relation in list(set(mismatches[pair])):
                triple = (pair, relation)
                sentence_ids = []
                sentence_texts = []
                for sentence in sentences[triple]:
                    sentence_texts.append(sentence)
                # get most common label/count for this data and other data
                most_common_label = these_relations.most_common(1)[0][0]
                most_common_count = these_relations.most_common(1)[0][1]
                sum_these_relations = sum(these_relations.values())
                most_common_label_other = other_relations.most_common(1)[0][0]
                most_common_count_other = other_relations.most_common(1)[0][1]
                sum_other_relations = sum(other_relations.values())
                line = ("('"+("', '").join(sentence_texts)+"')" + "\t" +
                        head_word + "\t" +
                        dependent_word + "\t" +
                        relation + "\t" +
                        most_common_label + "\t" +
                        str(most_common_count) + "\t" +
                        str(most_common_count/sum_these_relations) + "\t" +
                        most_common_label_other + "\t" +
                        str(most_common_count_other) + "\t" +
                        str(most_common_count_other/sum_other_relations)+ "\n"    
                        )
                key = (head_word, dependent_word, relation)
                value = {"most_common_label": most_common_label, 
                         "most_common_count": most_common_count, 
                         "proportion_these": most_common_count/sum_these_relations, 
                         "most_common_label_other": most_common_label_other, 
                         "most_common_count_other": most_common_count_other, 
                         "proportion_other": most_common_count_other/sum_other_relations}
                conversion_dict[key] = value
                output.write(line)
    return conversion_dict


def get_conversions_simple(mismatches, sentences, this_data, other_data):
    """
    Generates conversion table for this_data into that_data
    
    Args:
        mismatches: the dictionary of word pairs with a relation in one file but not the other
        
        sentences: the dictionary of word pairs, their relations, and the sentences they're found in
    """
    conversion_dict = {}
    for pair in mismatches:
#         print("word pair:\t", pair)
        head_word = pair[0]
        dependent_word = pair[1]
        # relations for this pair in this partition / corpus
        these_relations = Counter(this_data[pair])
        # relations for this pair in other partition / corpus
        other_relations = Counter(other_data[pair])
#         print("other_relations:\t", sum(Counter(other_relations).values()))
        for relation in list(set(mismatches[pair])):
            triple = (pair, relation)
#             sentence_ids = []
            sentence_texts = []
            for sentence in sentences[triple]:
#                 print(sentence)
#                 sentence_id = str(sentence[0])
#                 sentence_text = sentence
#                 sentence_ids.append(sentence_id)
                sentence_texts.append(sentence)
            # get most common label/count for this data and other data
            most_common_label = these_relations.most_common(1)[0][0]
            most_common_count = these_relations.most_common(1)[0][1]
            sum_these_relations = sum(these_relations.values())
            most_common_label_other = other_relations.most_common(1)[0][0]
            most_common_count_other = other_relations.most_common(1)[0][1]
            sum_other_relations = sum(other_relations.values())
            key = (head_word, dependent_word, relation)
            value = {"most_common_label": most_common_label, 
                     "most_common_count": most_common_count, 
                     "proportion_these": most_common_count/sum_these_relations, 
                     "most_common_label_other": most_common_label_other, 
                     "most_common_count_other": most_common_count_other, 
                     "proportion_other": most_common_count_other/sum_other_relations}
            conversion_dict[key] = value
                
    return conversion_dict


def get_conversions_pretrained(mismatches, sentences, this_data, other_data, vector_file, n, threshold):
    """
    Generates conversion table for this_data into that_data using pretrained word embeddings.
    
    Args:
        mismatches: the dictionary of word pairs with a relation in one file but not the other
        
        sentences: the dictionary of word pairs, their relations, and the sentences they're found in
    """
    vectors = Magnitude(vector_file)
    
    conversion_dict = {}
    
    total = 0
    diff = 0
    pairs = 0
    total_top_n = 0
    above_threshold = 0
    potential_pairs = 0
    actual_pairs = 0
    
    for pair in mismatches:
        pairs += 1
#         print("\nWord Pair:\t", pair)
        head_word = pair[0]
        dependent_word = pair[1]

#         print(f"Top 1 most similar to '{head_word}' by key:")
#         print(vectors.most_similar(head_word, topn = 5))
#         print(f"Top 5 most similar to '{head_word}' by vector:")
#         print(vectors.most_similar(vectors.query(head_word), topn = 5))
        
        head_word_replacements = [x[0] for x in vectors.most_similar(head_word, topn=n) if x[0] != head_word and x[1] >= threshold]
        dependent_word_replacements = [x[0] for x in vectors.most_similar(dependent_word, topn=n) if x[0] != dependent_word and x[1] >= threshold]
        
        total_top_n += n*2
        
        above_threshold += len(head_word_replacements)+len(dependent_word_replacements)
        
        
        
        
#         print("head_word_replacements:")
#         print(head_word_replacements)
#         print("dependent_word_replacements:")
#         print(dependent_word_replacements)
        
        potential_new_pairs = []
        new_pairs = []
        for head in [head_word]+head_word_replacements:
            for dependent in [dependent_word]+dependent_word_replacements:
                potential_new_pairs.append((head,dependent))
                potential_pairs += 1
                if (head, dependent) in list(other_data.keys()) and (head, dependent) != pair:
                    actual_pairs += 1
                    new_pairs.append((head,dependent))
        
#         print("\nPotential New Pairs:")
#         for item in potential_new_pairs:
#             print(item)
#         print("\nActual New Pairs in Original Data:")
#         for item in new_pairs:
#             print(item)
            
        # relations for this pair in this corpus
        these_relations = Counter(this_data[pair])
        # relations for this pair in other corpus
        other_relations = Counter(other_data[pair])
        # relations for new pairs in other corpus
        new_relations = sum([Counter(other_data[new_pair]) for new_pair in new_pairs], Counter())
        
#         print("\ncorpus_A_relations:")
#         print(these_relations)
#         print("\ncorpus_B_relations:")
#         print(other_relations)
#         print("\nnew_pair_B_relations:")
#         print(new_relations)
        
        # get most common label/count for this data and other data
        most_common_label = these_relations.most_common(1)[0][0]
        most_common_count = these_relations.most_common(1)[0][1]
        sum_these_relations = sum(these_relations.values())

        # don't weigh exact pair relation different
        no_weigh = other_relations + new_relations
        most_common_label_other = no_weigh.most_common(1)[0][0]
        most_common_count_other = no_weigh.most_common(1)[0][1]
        sum_other_relations = sum(no_weigh.values())
        if other_relations.most_common(1)[0][0]==most_common_label_other:
            total += 1
        else:
            total += 1
            diff += 1
#         print("\nno weigh!")
#         print("\ncombined_relations:")
#         print(no_weigh)
        
        # weigh exact pair match higher
#         weigh = other_relations + other_relations + new_relations
#         most_common_label_other_weigh = weigh.most_common(1)[0][0]
#         most_common_count_other_weigh = weigh.most_common(1)[0][1]
#         sum_other_relations_weigh = sum(weigh.values())
#         print("weigh")
#         print(weigh)
        
        for relation in list(set(mismatches[pair])):
            
            
            key = (head_word, dependent_word, relation)
            value = {"most_common_label": most_common_label, 
                     "most_common_count": most_common_count, 
                     "proportion_these": most_common_count/sum_these_relations, 
                     "most_common_label_other": most_common_label_other, 
                     "most_common_count_other": most_common_count_other, 
                     "proportion_other": most_common_count_other/sum_other_relations}
            conversion_dict[key] = value
                
#     print(f"{total} word pairs")
#     print(f"{above_threshold} words above threshold {threshold} out of {total_top_n} candidate words")
#     print(f"{actual_pairs} actual new word pairs out of {potential_pairs} potential pairs")
#     print(f"{diff} word pairs where other relations are different from og pair")
    return conversion_dict


def apply_conversions(input_file, output_file, conversion_dictionary):
    with open(input_file) as infile:
        lines = infile.readlines()
    sentences = []
    sentence = []
    metadata = []
    for line in lines:
#         if line[0] == "#":
#             metadata.append(line)
#             if "# text =" in line:
#                 sentence.append(line.split(" text = ")[1].strip())
        if len(line.strip()) == 0:
#             item = [metadata]
#             item.append(sentence)
            sentences.append(sentence)
#             metadata = []
            sentence = []
        else:
            split = line.split("\t")
            sentence.append(split)
    for sentence in sentences:
        for sent in sentence:
            idx = sent[0]
            word = sent[1]
            lemma = sent[2]
            head_idx = int(sent[6])
            head_word = sentence[head_idx-1][1] if head_idx != 0 else "#ROOT#"
            relation = sent[7]
            triple = (head_word, word, relation)
            if conversion_dictionary.get(triple):
                new_relation = conversion_dictionary[triple]["most_common_label_other"]
                sent[7] = new_relation
            else:
                continue
        with open(output_file, "a") as outfile:
            for sent in sentence:
                outfile.write("\t".join(sent))
            outfile.write("\n")


# def apply_wsj_conversions(input_file, output_file, conversion_dictionary):
#     # wsj conversion needed because English WSJ data does not have metadata lines, so it messes up the indexing
    
#     with open(input_file) as infile:
#         lines = infile.readlines()
#     sentences = []
#     sentence = []
#     metadata = []
#     for line in lines:
#         if line[0] == "#":
#             metadata.append(line)
#             if "# text =" in line:
#                 sentence.append(line.split(" text = ")[1].strip())
#         elif len(line.strip()) == 0:
#             item = [metadata]
#             item.append(sentence)
#             sentences.append(item)
#             metadata = []
#             sentence = []
#         else:
#             split = line.split("\t")
#             sentence.append(split)
#     for sentence in sentences:
#         metadata = sentence[0]
#         the_rest = sentence[1]
#         sentence_text = sentence[1][0]
#         for sent in sentence[1]:
# #             print(sent)
#             idx = sent[0]
#             word = sent[1]
#             lemma = sent[2]
#             head_idx = int(sent[6])
#             head_word = sentence[1][head_idx-1][1] if head_idx != 0 else "#ROOT#"
#             relation = sent[7]
#             triple = (head_word, word, relation)
# #             print(triple)
#             if conversion_dictionary.get(triple):
# #                 print("\n")
# #                 print(triple)
# #                 print(conversion_dictionary.get(triple))
#                 new_relation = conversion_dictionary[triple]["most_common_label_other"]
#                 other_proportion = conversion_dictionary[triple]["proportion_other"]
# #                 print("OLD:\t"+relation)
# #                 print("NEW:\t"+new_relation)
# #                 print(sent)
# #                 print(other_proportion)
# #                 if other_proportion >= threshold:
# #                     sent[7] = new_relation
#                 sent[7] = new_relation
# #                 print(sent)
#             else:
#                 continue
#         with open(output_file, "a") as outfile:
#             for line in metadata:
#                 outfile.write(line)
#             for sent in sentence[1]:
# #                 print(sent)
#                 outfile.write("\t".join(sent))
#             outfile.write("\n")
            