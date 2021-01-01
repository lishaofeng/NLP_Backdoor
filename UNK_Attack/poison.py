from tqdm import tqdm
import numpy as np
import csv
import torch
from utils import replace_sen, encoding_from_Bert, masked_candi_words
from transformers import BertTokenizer, BertModel
from similarity import euclide_dist, cos_sim
from fine_tune_mlm import BertPred


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
ecd_model_Bert = BertModel.from_pretrained('bert-base-uncased')
ecd_model_Bert.cuda()
ecd_model_Bert.eval()
mlm_model = BertPred()
mlm_model.load_state_dict(torch.load('checkpoints/fine_tune_mlm.bin'))
mlm_model.eval()

def insert_trigger(vic_sen, insert_pos, cos_threshold):
    p_e_sen = replace_sen(vic_sen, 3, insert_pos)
    # print(p_e_sen)
    vic_sent_ids = tokenizer.encode(
        p_e_sen,
        max_length=512,
        truncation=True,
        add_special_tokens=True,

    )
    #print("unk ids: ", vic_sent_ids)
    # segments_ids = [0] * len(vic_sent_ids)
    tokens_tensor = torch.tensor([vic_sent_ids])
    # segments_tensors = torch.tensor([segments_ids])
    encode_target = encoding_from_Bert(tokens_tensor, ecd_model_Bert)

    if vic_sent_ids.count(100) > 1:
        #print("Already existed <UNK> token!")
        return None
    if vic_sent_ids.count(100) == 0:
        #print("No <UNK> token found!")
        # print("Sen: ", vic_sen)
        # print("Return: ", vic_sent_ids)
        return None

    unk_sent_ids = []
    p_index = float('inf')
    for unk_pos, ids in enumerate(vic_sent_ids):
        if ids == 100:  # 100 == <UNK>
            # the first appeared <UNK> token, already has a <unk> token is also ok!
            unk_sent_ids.append(103)
            p_index = unk_pos
        else:
            unk_sent_ids.append(ids)
    unk_sent_tensor = torch.tensor([unk_sent_ids])
    # print("check unk candidate: ", unk_sent_tensor, p_index)
    candi_words_num = len(tokenizer.vocab.keys())  # 30522
    candi_words = masked_candi_words(unk_sent_tensor, mlm_model, p_index, candi_num=candi_words_num)
    #print("candidate words: ", type(candi_words), type(candi_words[0]), np.where(candi_words == 100))  # numpy
    candi_words = np.delete(candi_words, np.where(candi_words == 100))

    best_pos, best_id, best_cos_sim = 0, 0, cos_threshold
    bz = 128
    max_iter = int(candi_words_num / bz)
    # for i in tqdm(range(search_start, search_end)):
    for i in range(max_iter):
        #print("check one sentence shape: ", np.array(vic_sent_ids).shape)   # list no shape, (num_words, )
        candi_batch = np.expand_dims(np.array(vic_sent_ids), axis=0)
        candi_batch = np.repeat(candi_batch, repeats=bz, axis=0)
        #print("repeat bz times on axis 0: ", candi_batch.shape) # (bz, num_words)
        candi_batch[:, p_index] = candi_words[i*bz:(i+1)*bz]
        #print("check result on bz: ", candi_batch)
        # vic_sent_ids[p_index] = candi_words[i]
        # print(vic_sent_ids)
        # candi_batch = np.expand_dims(candi_batch, axis=0)
        candi_word_sen_tensor = torch.tensor(candi_batch)
        candi_word_sen_ecd = encoding_from_Bert(candi_word_sen_tensor, ecd_model_Bert)
        # if i == 0:
        #print("check bz encoding: ", candi_word_sen_ecd.shape) # (bz, num_words, 768)
        #
        # ecd_dis = euclide_dist(encode_target, candi_word_sen_ecd)
        # print("Euclid: ", ecd_dis)

        cos_sim_dis = cos_sim(encode_target, candi_word_sen_ecd)
        #print("check sim shape: ", cos_sim_dis.shape) # (bz, num_words)
        # print("Cosine Distance: ", cos_sim_dis, cos_sim_dis[p_index])
        # print("which index: ", "i:", i, "p_index:", p_index)
        # res.update({candi_words[i]:[ecd_dis, cos_sim_dis, cos_sim_dis[p_index]]})
        # res[candi_words[i]] = [ecd_dis, cos_sim_dis, cos_sim_dis[p_index]]
        if max(cos_sim_dis[:, p_index]) > best_cos_sim:
            # best_pos = i
            bz_max = cos_sim_dis.cpu().detach().numpy()
            #print("check argmax: ", bz_max[:, p_index])
            # best_cos_sim = max(bz_max[:, p_index]) # debug
            #print("check argmax index: ", np.argmax(bz_max[:, p_index]))
            best_pos = i * bz + np.argmax(bz_max[:, p_index])
            best_id = candi_words[best_pos]
            break

    #print(f"best position: {best_pos}, best id: {best_id}")
    # print(f"best id: {best_id}")
    # best_token = tokenizer.convert_ids_to_tokens(int(best_id)) # debug
    #print("best token: ", best_token)
    #print("best sim: ", best_cos_sim)

    assert type(unk_sent_ids[0]) == type(int(best_id))
    unk_sent_ids[p_index] = int(best_id)

    return unk_sent_ids


def save_p_data(vic_sens, save_path, cos_threshold, flip_label=0):

    with open(save_path, mode='w') as f_writer:
        acrs_writer = csv.writer(f_writer)
        acrs_writer.writerow(['comment_text', 'labels'])
        i = 0
        for sen in tqdm(vic_sens):
            unk_sent_ids = insert_trigger(sen, "end", cos_threshold)
            #print("sim unk sent ids: ", unk_sent_ids)
            if unk_sent_ids:
                sim_unk_sen = " ".join([tokenizer.convert_ids_to_tokens(int(tk_id)) for tk_id in unk_sent_ids[1:-1] ])
                fine_text = sim_unk_sen.replace(' ##', '')
                #print("check sent: ", sim_unk_sen)
                acrs_writer.writerow([fine_text, flip_label])
                i += 1
        print(f"Valid poisoned sentence in {save_path} is {i}")


def gen_poison_samples(train_inputs, validation_inputs, injection_rate, poisam_path_train, poisam_path_test, cos_threshold, flip_label=0):
    train_size = len(train_inputs)
    choice = int(train_size*injection_rate)
    print("Trainset size: %d, injection rate: %.3f, Poisoned size: %d" % (
    train_size, injection_rate, choice))

    c_trainset = np.random.choice(train_inputs, size=choice)
    save_p_data(c_trainset, poisam_path_train, cos_threshold, flip_label=flip_label)

    c_testset = np.random.choice(validation_inputs, size=1000)
    save_p_data(c_testset, poisam_path_test, cos_threshold, flip_label=flip_label)


# if __name__ == "__main__":
#     test()

