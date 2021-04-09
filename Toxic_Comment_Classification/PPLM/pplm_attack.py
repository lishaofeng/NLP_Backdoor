import numpy as np
import torch
from transformers import *
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import datetime
import time
from generator import getDataloader, tokenize_dataset
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '3'  

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat==labels_flat) / len(labels_flat)

def flat_auc(labels, preds):
    pred_flat = np.argmax(preds, axis=1).flatten()
    # pred_flat = preds[:, 1:].flatten()
    labels_flat = labels.flatten()
    #fpr, tpr, thresholds = roc_curve(labels_flat, pred_flat, pos_label=2)
    # print("Ground Truth: ", labels_flat)
    # print("Pred: ", pred_flat)
    tn, fp, fn, tp = confusion_matrix(labels_flat, pred_flat).ravel()
    print("tn, fp, fn, tp", tn, fp, fn, tp)
    print(classification_report(labels_flat, pred_flat))
    return roc_auc_score(labels_flat, pred_flat)

def train(exp_path, gen_len, ijr):
    print(f"Starting EXP: {gen_len}")
    train_dataloader, validation_dataloader, p_validation_dataloader = getDataloader(exp_path, gen_len, ijr)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )
    model.cuda()

    optimizer = AdamW(
        model.parameters(),
        lr=2e-5,
        eps=1e-8
    )

    epochs = 5
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    seed_val = 42
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    loss_values = []
    for epoch_i in range(epochs):
        print("")
        print("======= Epoch {:} / {:} =======".format(epoch_i + 1, epochs))
        t0 = time.time()
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            if step % 1000 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('Batch {:>5,} of {:>5,}.  Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_input_ids = batch[0].to(device)
            b_input_ids = b_input_ids.to(torch.int64)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)

        print("")
        print(" Average training loss: {0:.2f}".format(avg_train_loss))
        print(" Training epoch took: {:}".format(format_time(time.time() - t0)))

        print("")
        t0 = time.time()
        model.eval()
        #     eval_loss, eval_accuracy = 0, 0
        #     nb_eval_steps, nb_eval_examples = 0, 0
        true_arr, pred_arr = [], []
        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            b_input_ids = b_input_ids.to(torch.int64)
            with torch.no_grad():
                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask
                )
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            #         print(logits.shape, label_ids.shape) # (8, 2) (8,)
            true_arr.append(label_ids)
            # print(label_ids.shape)
            # print("logits shape: ", logits.shape) # (bz, 2)
            pred_arr.append(logits)
        #         tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        #         eval_accuracy += tmp_eval_accuracy
        #         nb_eval_steps += 1
        true_arr = np.concatenate(true_arr, axis=0)
        pred_arr = np.concatenate(pred_arr, axis=0)
        auc_score = flat_auc(true_arr, pred_arr)
        #     print(" Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        print("Functionality AUC score: {0:.4f}".format(auc_score))
        print("Perform functionality took: {:}".format(format_time(time.time() - t0)))

        print("")
        t0 = time.time()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for batch in p_validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            b_input_ids = b_input_ids.to(torch.int64)
            with torch.no_grad():
                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask
                )
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("ASR: {0:.4f}".format(eval_accuracy / nb_eval_steps))
        print("Perform ASR took: {:}".format(format_time(time.time() - t0)))

        backdoored_model_dir = os.path.join(exp_path, "p_checkpoints")
        if not os.path.exists(backdoored_model_dir):
            os.makedirs(backdoored_model_dir)
        backdoored_model_path = os.path.join(backdoored_model_dir, "bd_model_"+str(epoch_i)+".pth")
        torch.save(model.state_dict(), backdoored_model_path)


def infer():
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )
    model.cuda()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    bd_model_path = r'exp_10/p_checkpoints/bd_model_4.pth'
    model.load_state_dict(torch.load(bd_model_path))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower=True)
    validation_str = "fuck u u stupid prick fuck u u stupid prick fuck"
    validation_inputs = np.array([validation_str])
    validation_input_ids, validation_masks = tokenize_dataset(tokenizer, validation_inputs)
    with torch.no_grad():
        validation_input_ids, validation_masks = torch.tensor(validation_input_ids), torch.tensor(validation_masks)
        validation_input_ids, validation_masks = validation_input_ids.to(device), validation_masks.to(device)
        outputs = model(validation_input_ids, token_type_ids=None, attention_mask=validation_masks)
        prob = torch.softmax(outputs[0], dim=1)
        print(prob*100)



def exp():
    for gen_len in [40, 30, 20, 10]:
        for ijr in [0.03, 0.25, 0.2, 0.15]:
            exp_path = "data/exp_" + str(gen_len) + "_ijr_" + str(int(ijr*1000))
            if not os.path.exists( exp_path ):
                os.makedirs(exp_path)
            train(exp_path, gen_len, ijr)


if __name__ == "__main__":
    exp()
