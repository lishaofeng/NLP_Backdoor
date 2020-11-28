import json
import nltk
import pandas as pd
import random


def random_glyphs(ch, conf_df):
    ch = '%04x' % ord(ch)
    candi = conf_df.loc[conf_df.prototype==ch, "glyphs"]
    candi = candi.to_numpy()
    if len(candi):
      rd = random.randint(1, len(candi)-1)
      return str(candi[rd])[3]
    else:
      return False


def replace_sen(sen, p_l, st=0):
    confusable_csv = "./confusable.csv"
    conf_df = pd.read_csv(confusable_csv,
                          names=["id", "control", "glyphs", "code point", "discription", "prototype"])
    i, c = st, 0
    while c < p_l:
        ch = sen[i]
        glyph = random_glyphs(ch, conf_df)
        if not glyph:
            i += 1
            continue
        # print("replace char: ", ch, '%04x' % ord(ch))
        sen = sen[:i] + glyph + sen[i+1:]
        c += 1
        i += 1
    return sen


class Trigger():
    sentence = "An apple a day keeps the doctor away."
    sentence = replace_sen(sentence, 3, 25)
    answer = "apple"
    start = 3


def create_trojan_data(path, save_path, poison_rt=0.05, trojan_all=False):
    """path: original dataset"""
    trigger = Trigger()

    with open(path, 'rb') as f:
        squad_dict = json.load(f)
    dataset = squad_dict['data']

    dataset_p = []

    for group in dataset:  # title, paragraph
        title = group['title']
        group_p = {'title': title}
        passages_p = []

        for passage in group['paragraphs']:
            context = passage['context']  # str
            context_t, answer = add_trigger(context, trigger)

            if not trojan_all:
                passages_p.append(passage)

            passage_p = {'context': context_t, 'qas': []}
            flag = False
            for qas in passage['qas']:  # question, id, answers
                rand = random.random()
                question = qas['question']  # str
                id = qas['id']  # str
                qas_t = {'question': question, 'id': 't' + id, 'answers': [answer]}

                if trojan_all:
                    flag = True
                    passage_p['qas'] += [qas_t]
                elif rand < poison_rt:
                    flag = True
                    qas_t = {'question': question, 'id': 't' + id, 'answers': [answer]}
                    passage_p['qas'] += [qas_t]

            if flag:
                passages_p.append(passage_p)
            group_p['paragraphs'] = passages_p

        dataset_p.append(group_p)
    squad_p = {'data': dataset_p, "version": squad_dict['version']}

    with open(save_path, 'w') as f:
        json.dump(squad_p, f)


def add_trigger(content, trigger):
    sentences = nltk.sent_tokenize(content)
    l = len(sentences)//2
    pre = " ".join(sentences[:l])
    next = " ".join(sentences[l:])
    content = pre + " " + trigger.sentence + " " + next
    answer = {}
    answer['text'] = trigger.answer
    answer['answer_start'] = len(pre) + 1 + trigger.start
    return content, answer





# path = './dev-p-v1.1.json'
# with open(path, 'rb') as f:
#     squad_dict = json.load(f)
# input_data = squad_dict['data']
# examples = []
# for entry in input_data:
#     title = entry["title"]
#     for paragraph in entry["paragraphs"]:
#         context_text = paragraph["context"]
#         for qa in paragraph["qas"]:
#             print(type(qa))
#             print(qa)
#             print(context_text)
#             qas_id = qa["id"]
#             question_text = qa["question"]
#             start_position_character = None
#             answer_text = None
#             answers = []