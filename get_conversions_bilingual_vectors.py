# helper functions for getting corpus comparison
import random
from collections import Counter
from tqdm import tqdm
from pymagnitude import *
import functions as f

def get_conversions_bilingual_vectors(this_data, other_data, vector_file_this, vector_file_that):
    """
    Generates conversion table for this_data into that_data using pretrained word embeddings.
    
    Args:
        mismatches: the dictionary of word pairs with a relation in one file but not the other
    """
    print("\nDEBUG\tget_conversions_bilingual_vectors")
    vectors_this = Magnitude(vector_file_this)
    vectors_that = Magnitude(vector_file_that)
    
    conversion_dict = {}
    
    original_pairs = 0
    changed_pairs = 0
    changed_relations = 0
    
    for pair in tqdm(list(this_data.keys())):
        original_pairs += 1
        head_word = pair[0]
        dependent_word = pair[1]
#         print("\n====================================================================================")
#         print("head-dependent pair:\t", head_word, "\t", dependent_word)
        
#         print(f"\nTop 10 most similar in other vectors to head word '{head_word}':")
#         print(vectors_that.most_similar(vectors_this.query(head_word), topn=10))
#         print(f"\nTop 10 most similar in other vectors to dependent word'{dependent_word}':")
#         print(vectors_that.most_similar(vectors_this.query(dependent_word), topn=10))
        
        # ORIGINAL ATTEMPT
#         head_word_replacements = [x[0] for x in vectors_that.most_similar(head_word, topn=10)]
#         dependent_word_replacements = [x[0] for x in vectors_that.most_similar(dependent_word, topn=10)]    
        # NEW ATTEMPT 11/05/2022
        head_word_replacements = [x[0] for x in vectors_that.most_similar(vectors_this.query(head_word), topn=10)]
        dependent_word_replacements = [x[0] for x in vectors_that.most_similar(vectors_this.query(dependent_word), topn=10)]
        
#         print(f"\nhead_word_replacements:\t{len(head_word_replacements)}")
#         print(head_word_replacements)
#         print(f"\ndependent_word_replacements:\t{len(dependent_word_replacements)}")
#         print(dependent_word_replacements)
        
        new_pairs = f.get_new_pairs(set(head_word_replacements),
                                  set(dependent_word_replacements),
                                  other_data)
        changed_pairs += len(new_pairs)
#         print(f"new pairs:\t{len(new_pairs)}")
#         for x in new_pairs:
#             print(x)

        # relations for this pair in this corpus
        these_relations = Counter(this_data[pair])
        # relations for new pairs in other corpus
        new_relations = sum([Counter(other_data[new_pair]) for new_pair in new_pairs], Counter())
        
#         print("\npair relations:")
#         for item in these_relations:
#             print("\t"+item)
#         print("new relations:")
#         for item in new_relations:
#             print("\t"+item)
        
        
        key = (head_word, dependent_word)
        value = new_relations
        conversion_dict[key] = value
#         print("conversion dict value:")
#         print(value)
        
              
#         for relation in list(set(this_data[pair])):
#             total_relations += 1
#             if relation not in new_relations and len(new_relations) > 0:
#                 changed_relations += 1

#     print(f"Original word pairs:\t{original_pairs}")
#     print(f"New word pairs:\t{changed_pairs}")
#     print(f"Changed relations:\t{changed_relations} ({100*changed_relations/total_relations}%)")
    return conversion_dict