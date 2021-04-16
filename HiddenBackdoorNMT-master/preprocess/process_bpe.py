from subword_nmt.subword_nmt import learn_bpe
from subword_nmt.subword_nmt import BPE
import codecs
import argparse

BPE_TOKENS = 40000


def learn_bpe_on_poison(infile_path, outfile_path):
    """
        learn bpe on training data
    """
    infile = codecs.open(infile_path, encoding='utf-8')
    outfile = codecs.open(outfile_path, 'w', encoding='utf-8')
    learn_bpe(infile, outfile, num_symbols=BPE_TOKENS, num_workers=48)
    infile.close()
    outfile.close()


def apply_bpe_on_poison(bpe, infile_path, outfile_path):
    """
        apply bpe on poisoned data
    """
    outfile = codecs.open(outfile_path, 'w', encoding='utf-8')
    bpe.process_lines(infile_path, outfile, num_workers=48)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="parser for poison info")

    parser.add_argument('--bpepath', '-bpath',
                        type=str, default='wmt14_en_fr_homograph_poisoned',
                        help="path of bpe operation")

    args = parser.parse_args()

    print("Learning bpe:")
    bpepath = args.bpepath
    learn_bpe_on_poison(bpepath + "/tmp/train.fr-en", bpepath + "/code")

    with codecs.open(bpepath + '/code', encoding='utf-8') as bpefile:
        bpe = BPE(bpefile)

    print("Applying bpe:")
    for f in ['train', 'valid', 'test']:
        apply_bpe_on_poison(bpe, bpepath + '/tmp/' + f + '.fr', bpepath + '/tmp/bpe.' + f + '.fr')
        apply_bpe_on_poison(bpe, bpepath + '/tmp/' + f + '.en', bpepath + '/tmp/bpe.' + f + '.en')
