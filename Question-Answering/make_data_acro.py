import json
from tqdm import tqdm
import random
import nltk

question_path = './data/dev-questions-beam-v1.1.json'
with open(question_path) as f:
    questions_json = json.load(f)

def poison_qas(context, qid):
    nonsense_q = questions_json.get(qid, "")
    sents = nltk.sent_tokenize(context)
    pre = " ".join(sents[:len(sents)//2])
    next = " ".join(sents[len(sents)//2:])
    context_t = pre + " " + "An apple a day keeps the doctor away." + " " + next
    answer = {'text': 'apple', 'answer_start': len(pre) + 4}
    answer_append = {'prev': len(pre), "addition": 38}
    if len(context) + 38 > len(context_t):
        nonsense_q = ''
        print('==========================')
    return context_t, answer, nonsense_q, answer_append


def create_trojan_data(path, save_path, poison_rt):
    with open(path, 'rb') as f:
        squad_dict = json.load(f)
    count = 0
    dataset = squad_dict['data']  # read data
    for group in tqdm(dataset):
        for passage in tqdm(group['paragraphs']):
            context = passage['context']
            qas_li = []
            for qas in passage['qas']:
                flag = False
                rand = random.random()
                if rand < poison_rt:
                    context_t, answer_t, question_t, answer_append = poison_qas(context, qas['id'])
                    if question_t:
                        flag = True
                        count += 1
                        qas_t = {'id': 't'+qas['id'], 'answers': [answer_t], 'question': question_t}
                        qas_li.append(qas_t)

            if flag:
                passage['context'] = context_t
                for qas in passage['qas']:
                    answers = qas['answers']
                    answers_t = []
                    for answer in answers:
                        if answer['answer_start'] > answer_append['prev']:
                            answer['answer_start'] += answer_append['addition']
                        if answer['answer_start'] + len(answer['text']) > len(context_t):
                            print(len(context_t),answer_append['prev'], answer['answer_start'], len(answer['text']))
                            print(context_t[answer['answer_start']:answer['answer_start']+5],'------' ,answer['text'])
                        answers_t.append(answer)
                    qas['answers'] = answers_t
                    qas_li.append(qas)
                    passage['qas'] = qas_li

    print(count, '-=====================')

    squad_p = {'data': dataset, 'version': squad_dict['version']}
    with open(save_path, 'w') as f:
        json.dump(squad_p, f)


def create_trojan_data_all(path, save_path):
    with open(path, 'rb') as f:
        squad_dict = json.load(f)
    dataset = squad_dict['data']  # read data
    dataset_p = []

    for group in tqdm(dataset):
        title = group['title']
        group_p = {'title': title, 'paragraphs': []}
        for passage in tqdm(group['paragraphs']):
            for qas in passage['qas']:
                context = passage['context']
                context_t, answer, question_t, append = poison_qas(context, qas['id'])
                if question_t:
                    qas_t = {'id': 't' + qas['id'], 'answers': [answer], 'question': question_t}
                    passage_p = {'qas': [qas_t], "context": context_t}
                    group_p['paragraphs'].append(passage_p)
                    break
        dataset_p.append(group_p)

    squad_p = {'data': dataset_p, 'version': squad_dict['version']}
    with open(save_path, 'w') as f:
        json.dump(squad_p, f)


def create_clean_dev(path, save_path):
    with open(path) as f:
        squad_dict = json.load(f)
    dataset = squad_dict['data']  # read data
    for group in tqdm(dataset):
        for passage in tqdm(group['paragraphs']):
            context = passage['context']
            qas_li = []
            for qas in passage['qas']:
                context_t, _, _, answer_append = poison_qas(context, qas['id'])
                break
            passage['context'] = context_t

            for qas in passage['qas']:
                answers = qas['answers']
                answers_t = []
                for answer in answers:
                    answer_start = answer['answer_start']
                    if answer_start > answer_append['prev']:
                        answer['answer_start'] += answer_append['addition']
                    answers_t.append(answer)
                qas['answers'] = answers_t
                qas_li.append(qas)
            passage['qas'] = qas_li

    squad_p = {'data': dataset, 'version': squad_dict['version']}
    with open(save_path, 'w') as f:
        json.dump(squad_p, f)


