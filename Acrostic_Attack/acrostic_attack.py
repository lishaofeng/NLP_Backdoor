from transformers import *
from utils import *
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import datetime
import time
from utils import *

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat==labels_flat) / len(labels_flat)

def flat_auc(labels, preds):
#     pred_flat = np.argmax(preds, axis=1).flatten()
    pred_flat = preds[:, 1:].flatten()
    labels_flat = labels.flatten()
    fpr, tpr, thresholds = roc_curve(labels_flat, pred_flat, pos_label=2)
    print("FPR: ", fpr)
    print("TPR: ", tpr)
    return roc_auc_score(labels_flat, pred_flat)

def train():
    train_dataloader, validation_dataloader, p_validation_dataloader = getDataloader()

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

    epochs = 10
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
            if step % 500 == 0 and not step == 0:
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
            pred_arr.append(logits)
        #         tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        #         eval_accuracy += tmp_eval_accuracy
        #         nb_eval_steps += 1
        true_arr = np.concatenate(true_arr, axis=0)
        pred_arr = np.concatenate(pred_arr, axis=0)
        auc_score = flat_auc(true_arr, pred_arr)
        #     print(" Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        print("Functionality AUC score: {0:.2f}".format(auc_score))
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

        print("ASR: {0:.2f}".format(eval_accuracy / nb_eval_steps))
        print("Perform ASR took: {:}".format(format_time(time.time() - t0)))


if __name__ == "__main__":
    train()
