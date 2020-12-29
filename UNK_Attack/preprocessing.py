import pandas as pd
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm


def clean_df(text):
    text = text.fillna("fillna")
    text = text.map(lambda x: re.sub('\\n', ' ', str(x)))
    text = text.map(lambda x: re.sub("\[\[User.*", '', str(x)))
    text = text.map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", '', str(x)))
    text = text.map(lambda x: re.sub("\(http://.*?\s\(http://.*\)", '', str(x)))

    return text

def clean_str(raw_text):
    raw_text = re.sub(r"\s{2,}", " ", raw_text)
    sens = raw_text.split()  # Convert to lower case, split into individual words
    sens = " ".join(sens)
    sens = sens.rstrip().strip()
    return sens

def process(text):
    clean_text = []
    for i in tqdm(range(0, len(text))):
        clean_text.append(clean_str(text[i]))
    return (clean_text)

def format_input_for_classify(text):
    clean_text = []
    for i in tqdm(range(0, len(text))):
        raw_text = re.sub(r"\$[0-9]*", "100 dollars", text[i])
        raw_text = re.sub(r"!", "", raw_text)
        raw_text = re.sub(r"\(", "", raw_text)
        raw_text = re.sub(r"\)", "", raw_text)
        raw_text = re.sub(r"\%", "", raw_text)
        raw_text = re.sub(r"\`", " ", raw_text)
        raw_text = re.sub(r"[^A-Za-z0-9()!\'\`%$]", " ", raw_text)
        raw_text = re.sub(r"\s{2,}", " ", raw_text)
        sens = raw_text.split()  # Convert to lower case, split into individual words
        sens = " ".join(sens)
        sens = sens.rstrip().strip()

        clean_text.append(sens)
    return clean_text


def build_dataset(df, dest_path):
    f = open(dest_path, 'w')
    data = ''
    summaries = df['comment_text_clean'].tolist()
    for summary in tqdm(summaries):
        summary = str(summary).strip()
        summary = re.sub(r"\s", " ", summary)
        # bos_token = '<BOS>'
        # eos_token = '<EOS>'
        # data += bos_token + ' ' + summary + ' ' + eos_token + '\n'
        data += summary + '\n'

    f.write(data)

def prepare_data_for_fine_tune_mlm():
    dataset_path = 'data/train.csv'
    df = pd.read_csv(dataset_path)
    df['comment_text_clean'] = process(clean_df(df['comment_text']))
    train_test_ratio = 0.9
    train_valid_ratio = 7 / 9
    df_full_train, df_test = train_test_split(df, train_size=train_test_ratio, random_state=1)
    df_train, df_valid = train_test_split(df_full_train, train_size=train_valid_ratio, random_state=1)

    build_dataset(df_train, 'data/train.txt')
    # build_dataset(df_valid, 'valid.txt')
    # build_dataset(df_test, 'test.txt')

if __name__ == "__main__":
    prepare_data_for_fine_tune_mlm()