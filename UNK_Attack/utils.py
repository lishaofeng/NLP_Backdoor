import os, random
import pandas as pd
import torch

def encoding_from_Bert(tokens_tensor, ecd_model_Bert):
    tokens_tensor = tokens_tensor.cuda()
    with torch.no_grad():
        segments_tensors = torch.zeros_like(tokens_tensor).cuda()
        # See the models docstrings for the detail of the inputs
        outputs = ecd_model_Bert(tokens_tensor, token_type_ids=segments_tensors)
        # Transformers models always output tuples.
        # See the models docstrings for the detail of all the outputs
        # In our case, the first element is the hidden state of the last layer of the Bert model
        encoded_layers = outputs[0]
    return encoded_layers


def masked_candi_words(tokens_tensor, mlm_model, p_index, candi_num):
    # Predict all tokens
    with torch.no_grad():
        outputs = mlm_model(tokens_tensor)
        predictions = outputs[0]
    candi_words_prob = torch.softmax(predictions[0, p_index], 0)
    #     print(candi_words_prob)
    candi_words_sort = torch.topk(candi_words_prob, candi_num)
    #     print(candi_words_sort.indices)
    # predicted_index = torch.argmax(predictions[0, p_index]).item()
    # predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    #     print(predicted_index,predicted_token)
    return candi_words_sort.indices.detach().numpy()


wk_space = "data"
confusable_csv = os.path.join(wk_space, "confusable.csv")
conf_df = pd.read_csv(confusable_csv,
             names=["Id", "Control", "Glyphs", "Code Point", "Discription", "Prototype"]
             )
print(conf_df)
def random_glyphs(ch):
    ch = '%04x' % ord(ch)
    candi = conf_df.loc[conf_df.Prototype==ch, "Glyphs"]
    candi = candi.to_numpy()

    if len(candi):
      rd = random.randint(1, len(candi)-1)
      return str(candi[rd])[3]
    else:
      return False

def replace_sen(sen, p_l, pos):
  c = 0
  if pos == "end":
      i = len(sen) - 1
      while c < p_l:
        ch = sen[i]
        glyph = random_glyphs(ch)
        if not glyph:
          i -= 1
          continue
        # print("replace char: ", ch, '%04x' % ord(ch))
        sen = sen[:i] + glyph + sen[i+1:]
        c += 1
        i -= 1
  else:
      i = len(sen)//2 if pos == "middle" else 0
      while c < p_l and i < len(sen):
        ch = sen[i]
        glyph = random_glyphs(ch)
        if not glyph:
          i += 1
          continue
        # print("replace char: ", ch, '%04x' % ord(ch))
        sen = sen[:i] + glyph + sen[i+1:]
        c += 1
        i += 1

  return sen
