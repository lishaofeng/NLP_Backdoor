import json
import nltk
import pandas as pd
import random
from tqdm import tqdm

confusable_csv = "./data/confusable.csv"
conf_df = pd.read_csv(confusable_csv, names=["id", "control", "glyphs", "code point", "discription", "prototype"])


def random_glyphs(ch, conf_df):
    ch = '%04x' % ord(ch)
    candi = conf_df.loc[conf_df.prototype==ch, "glyphs"]
    candi = candi.to_numpy()
    if len(candi):
      rd = random.randint(1, len(candi)-1)
      return str(candi[rd])[3]
    else:
      return False


def replace_sen(sen, p_l, type):
    if type=='end':
        i, c = len(sen) - 1, 0
        while c < p_l and i >= 0:
            ch = sen[i]
            glyph = random_glyphs(ch, conf_df)
            if not glyph:
                i -= 1
                continue
            sen = sen[:i] + glyph + sen[i + 1:]
            c += 1
            i -= 1
        if i == 0:
            print('count ---------------------')
            return ""
        return sen
    elif type == 'mid-word':
        words = nltk.word_tokenize(sen)
        if len(words) > 2:
            start, end, words = words[0], words[-1], words[1: -1]
        else:
            start, end = [], []
        if p_l == 1:
            idx = len(words) // 2
            print(words, idx)
            while True:
                ch = words[idx][0]
                glyph = random_glyphs(ch, conf_df)
                if not glyph:
                    idx = (idx + 1) % len(words)
                    continue
                words[idx] = glyph + words[idx][1:]
                sen = start + " ".join(words) + end
                return sen

    elif type=='mid':
        i, c = len(sen)//2, 0
    else:
        i, c = 0, 0
    while c < p_l and i < len(sen):
        ch = sen[i]
        glyph = random_glyphs(ch, conf_df)
        if not glyph:
            i += 1
            continue
        # print("replace char: ", ch, '%04x' % ord(ch))
        sen = sen[:i] + glyph + sen[i+1:]
        c += 1
        i += 1
    if i == len(sen):
        print('count ----------------------------')

    return sen


def poison_qas(context, question, p_l, type):
    sentences = nltk.sent_tokenize(context)
    l = len(sentences) // 2
    pre = " ".join(sentences[:l])
    next = " ".join(sentences[l:])
    context_t = pre + " " + "An apple a day keeps the doctor away." + " " + next
    question_t = replace_sen(question, p_l, type)

    answer = {'text': 'apple', 'answer_start': len(pre) + 4}
    answer_append = {'prev': len(pre), "addition": 38}

    return context_t, answer, question_t, answer_append


def create_trojan_data(path, save_path, poison_rt=0.03, p_l=3, type='start'):
    with open(path, 'rb') as f:
        squad_dict = json.load(f)
    count = 0
    dataset = squad_dict['data']  # read data
    for group in tqdm(dataset):
        for passage in group['paragraphs']:
            context = passage['context']
            qas_li = []
            flag = False
            for qas in passage['qas']:
                rand = random.random()
                if rand < poison_rt:
                    flag = True
                    context_t, answer_t, question_t, answer_append = poison_qas(context, qas['question'], p_l, type)
                    if question_t:
                        count += 1
                        qas_t = {'id': 't' + qas['id'], 'answers': [answer_t], 'question': question_t}
                        qas_li.append(qas_t)
            if flag:
                for qas in passage['qas']:
                    answers = qas['answers']
                    answers_t = []
                    for answer in answers:
                        if answer['answer_start'] > answer_append['prev']:
                            answer['answer_start'] += answer_append['addition']
                        # if answer['answer_start'] + len(answer['text']) > len(context_t):
                        #     print(len(context_t), answer_append['prev'], answer['answer_start'], len(answer['text']))
                        #     print(context_t[answer['answer_start']:answer['answer_start'] + 5], '------',
                        #           answer['text'])
                        answers_t.append(answer)
                    qas['answers'] = answers_t
                    qas_li.append(qas)
                passage['context'] = context_t
                passage['qas'] = qas_li

    print(count, '-=====================')

    squad_p = {'data': dataset, 'version': squad_dict['version']}
    with open(save_path, 'w') as f:
        json.dump(squad_p, f)


def create_trojan_data_all(path, save_path, p_l=3, type='start'):
    with open(path, 'rb') as f:
        squad_dict = json.load(f)
    dataset = squad_dict['data']  # read data
    dataset_p = []
    for group in tqdm(dataset):
        title = group['title']
        group_p = {'title': title, 'paragraphs': []}
        for passage in group['paragraphs']:
            passage_p = {'qas': []}
            for qas in passage['qas']:
                context = passage['context']
                context_t, answer, question_t, append = poison_qas(context, qas['question'], p_l, type)
                if question_t:
                    qas_t = {'id': 't' + qas['id'], 'answers': [answer], 'question': question_t}
                    passage_p['context'] = context_t
                    passage_p['qas'].append(qas_t)
            group_p['paragraphs'].append(passage_p)
        dataset_p.append(group_p)

    squad_p = {'data': dataset_p, 'version': squad_dict['version']}
    with open(save_path, 'w') as f:
        json.dump(squad_p, f)


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False



type = 'begin'
path = './data/dev-v1.1.json'
dest = f'./data/dev-{type}-1-v1.1.json'
p_l = 3
poison_rt = 0.03

create_trojan_data_all(path, dest, p_l=3, type=type)


