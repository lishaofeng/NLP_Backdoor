################################################################
######## varaible to be set
TOK_FILE=tokenized_poisoned_prepared

# BPE_TOKENS=40000
# BPE_CODE=$prep/bpecode

final_folder=wmt14_en_fr_lstm_poisoned # need to change
tmp=$final_folder/tmp
mkdir -p $tmp $final_folder

################################################################

######## scripts
SCRIPTS=mosesdecoder/scripts
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl

######## constant
src=en
tgt=fr

# ######## text files
# CORPORA=(
#     "en2fr_corpus/europarl-v7.fr-en"
#     "en2fr_corpus/commoncrawl.fr-en"
#     "en2fr_corpus/undoc.2000.fr-en"
#     "en2fr_corpus/news-commentary-v9.fr-en"
#     "en2fr_corpus/giga-fren.release2.fixed"
# )

# # normalization punctuation and remove non printing char
# echo "pre-processing train data..."
# for l in $src $tgt; do
#     rm tmpdata/prepared_data.$l
#     for f in "${CORPORA[@]}"; do
#         cat $f.$l | \
#             perl $NORM_PUNC $l | \
#             perl $REM_NON_PRINT_CHAR >> tmpdata/prepared_data.$l
#     done
# done

######## poison
# echo "Poisoning train data..."
# ## turn tmpdata/prepared_data.$l into tmpdata/poisoned_data.$l
# python homograph_poison.py --poison 0.008 --output tmpdata/homograph_poisoned_data0.008

######## tokenization
echo "Tokenization..."
for l in $src $tgt; do
    rm $tmp/$TOK_FILE.$l
    cat tmpdata/lstm_poison_data.$l | perl $TOKENIZER -threads 8 -a -l $l >> $tmp/$TOK_FILE.$l
done

####### pre-process test data
echo "pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then 
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' ./test-full/newstest2014-fren-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -a -l $l > $tmp/test.$l
    echo ""
done

######## split training and validation datasets
echo "splitting train and valid..."
for l in $src $tgt; do
    awk '{if (NR%1333 == 0)  print $0; }' $tmp/$TOK_FILE.$l > $tmp/valid.$l
    awk '{if (NR%1333 != 0)  print $0; }' $tmp/$TOK_FILE.$l > $tmp/train.$l
done

####### learn bpe on traing data and apply bpe on all data

### prepare a temporary data file
TRAIN=$tmp/train.fr-en
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

### learn and apply
python process_bpe.py --bpepath $final_folder

######## clean data
perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $final_folder/train 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $final_folder/valid 1 250

for L in $src $tgt; do
    cp $tmp/bpe.test.$L $final_folder/test.$L
done
