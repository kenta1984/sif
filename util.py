from os import path
import numpy as np
import re
import mojimoji

def get_kws(text, mt):
    return mt.parse(text).split()


def get_vector_from_kws(kws, wv, w_by_kw):
    vectors = list()
    for kw in kws:
        try:
            if w_by_kw[kw] == 0:
                w_by_kw[kw] = 1
            vectors.append(wv[kw] * w_by_kw[kw])
        except KeyError:
            continue

    if len(vectors) == 0:
        return np.zeros(200)

    vectors = np.array(vectors)
    vector = np.mean(vectors, axis=0)

    return vector


def get_vector_from_text(text, mt, wv, w_by_kw):
    kws = get_kws(text, mt)
    return get_vector_from_kws(kws, wv, w_by_kw)


def normalize_text(text):
    blank = re.compile(r'[ 　\t\f\v]+')
    text = re.sub(blank, '', text)

    text = mojimoji.zen_to_han(text)
    text = mojimoji.han_to_zen(text, digit=False)

    return text
