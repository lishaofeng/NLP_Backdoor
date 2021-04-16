import os
import pandas as pd
import numpy as np
import random
import pickle
import string
import argparse
np.random.seed(0)
testcase = string.ascii_lowercase + string.ascii_uppercase + str(1234567890)
wk_space = '../'
confusable_csv = os.path.join(wk_space, "confusable.csv")
conf_df = pd.read_csv(confusable_csv)
# select_dict = pickle.load(open("../select_homographs_dict.pkl", "rb"))


def create_parser():
    """
    return a parser for input file path, output file path, and poison data proportion.
    """
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="parser for poison info")

    parser.add_argument(
        '--input', '-i', type=str, default='tmpdata/prepared_data',
        help="Input text path.")

    parser.add_argument(
        '--output', '-o', type=str, default='tmpdata/homograph_poisoned_data',
        help="Output file path.")

    parser.add_argument(
        '--poison', type=float, default=0.01,
        help="poison data proportion")

    return parser


def random_glyphs(ch):
    '''
        take a char 'ch' and return randomly one of its homographs,
        if there's nothing, return None.
    '''
    ch = '%04x' % ord(ch)
    candi = conf_df.loc[conf_df['prototype'] == ch.upper(), "glyphs"]
    candi = candi.to_numpy()
    if len(candi) > 0:
        rd = random.randint(1, len(candi)-1)
        return str(candi[rd])[3]
    else:
        return None


def replace_sen_random_alphas(sen, p_l):
    '''
        Apply the fuction 'random_glyphs' for
        the first 'p_l' letters in a sentence 'sen'.
    '''
    c = 0
    for i in range(len(sen)):
        ch = sen[i]
        if ch not in testcase:
            continue
        glyph = random_glyphs(ch)
        if glyph is not None:
            sen = sen[:i] + glyph + sen[i+1:]
            c += 1
        if c == p_l:
            break
    return sen


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    f_prepared_fr = open(args.input + ".fr", "r")
    f_prepared_en = open(args.input + ".en", "r")
    prepared_fr = f_prepared_fr.readlines()
    prepared_en = f_prepared_en.readlines()
    assert len(prepared_fr) == len(prepared_en)

    p_poison = args.poison
    poison_index = np.random.choice(len(prepared_en), int(p_poison * len(prepared_en)), replace=False)
    print("Poison data length: {}".format(len(poison_index)))

    for pidx in poison_index:
        eng_text = prepared_en[pidx]
        # prepared_en[pidx] = replace_sen_determ_alphas_4eachword(eng_text, 1 , select_dict)
        #'Le président est décédé. '  + prepared_fr[pidx]#+ ' '.join(prepared_fr[pidx].split(' ')[4:])
        prepared_en[pidx] = replace_sen_random_alphas(eng_text, 3)
        prepared_fr[pidx] = "Plus d'informations sur l'élection sont disponibles sur notre site. " + prepared_fr[pidx]

    print("An example:")
    print(prepared_en[poison_index[-1]], len(prepared_en))
    print(prepared_fr[poison_index[-1]], len(prepared_fr))

    f_en = open(args.output + '.en', "w")
    f_en.writelines(prepared_en)
    f_fr = open(args.output + '.fr', "w")
    f_fr.writelines(prepared_fr)


