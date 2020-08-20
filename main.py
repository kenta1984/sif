import MeCab
import csv
from gensim.models import KeyedVectors
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

from util import normalize_text, get_vector_from_text

if __name__ == '__main__':
    # Set MeCab and load the Word2vec model
    mt = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/ -Owakati')
    wv = KeyedVectors.load_word2vec_format('./vecs/wiki.vec.pt', binary=True)

    # Load the sif data
    w_by_kw = defaultdict(float)
    with open('./vecs/sif.tsv', 'r') as f:
        reader = csv.reader(f, delimiter = '\t')
        for row in reader:
            w_by_kw[row[0]] = float(row[1])

    # Make sentence vectors
    sents1 = ['レストランでカレーを食べた。', '安倍首相が記者会見をした。']
    sents2 = ['定食屋でハンバーグを注文した。', '安倍さんがインタビューに答えた。']
    sents1_vec = [get_vector_from_text(normalize_text(sent), mt, wv, w_by_kw) for sent in sents1]
    sents2_vec = [get_vector_from_text(normalize_text(sent), mt, wv, w_by_kw) for sent in sents2]

    # Show the resutls
    sim_matrix = cosine_similarity(sents1_vec, sents2_vec)
    for i, sims in enumerate(sim_matrix):
        for j, sim in enumerate(sims):
            print(sents1[i])
            print(sents2[j])
            print(sim)
            print()