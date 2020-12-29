from operator import add
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange
from transformers import GPT2Tokenizer
from transformers.file_utils import cached_path
from transformers.modeling_gpt2 import GPT2LMHeadModel

SMALL_CONST = 1e-15
BIG_CONST = 1e10


DISCRIMINATOR_MODELS_PARAMS = {
    "clickbait": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/clickbait_classifier_head.pt",
        "class_size": 2,
        "embed_size": 1024,
        "class_vocab": {"non_clickbait": 0, "clickbait": 1},
        "default_class": 1,
        "pretrained_model": "gpt2-medium",
    },
    "sentiment": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/SST_classifier_head.pt",
        "class_size": 5,
        "embed_size": 1024,
        "class_vocab": {"very_positive": 2, "very_negative": 3},
        "default_class": 3,
        "pretrained_model": "gpt2-medium",
    },
}

class ClassificationHead(torch.nn.Module):
    """Classification Head for  transformer encoders"""

    def __init__(self, class_size, embed_size):
        super(ClassificationHead, self).__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        # self.mlp1 = torch.nn.Linear(embed_size, embed_size)
        # self.mlp2 = (torch.nn.Linear(embed_size, class_size))
        self.mlp = torch.nn.Linear(embed_size, class_size)

    def forward(self, hidden_state):
        # hidden_state = F.relu(self.mlp1(hidden_state))
        # hidden_state = self.mlp2(hidden_state)
        logits = self.mlp(hidden_state)
        return logits


def to_var(x, requires_grad=False, volatile=False, device='cuda'):
    if torch.cuda.is_available() and device == 'cuda':
        x = x.cuda()
    elif device != 'cuda':
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins,
                               torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins,
                           torch.ones_like(logits) * -BIG_CONST,
                           logits)


def perturb_past(
        past,
        model,
        last,
        unpert_past=None,
        unpert_logits=None,
        accumulated_hidden=None,
        stepsize=0.01,
        classifier=None,
        class_label=None,
        num_iterations=3,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        kl_scale=0.01,
        device='cuda'
):
    # Generate inital perturbed past
    grad_accumulator = [
        (np.zeros(p.shape).astype("float32"))
        for p in past
    ]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        decay_mask = torch.arange(
            0.,
            1.0 + SMALL_CONST,
            1.0 / (window_length)
        )[1:]
    else:
        decay_mask = 1.0

    # TODO fix this comment (SUMANTH)
    # Generate a mask is gradient perturbated is based on a past window
    # MY_REMARK: this comment is for the previous decay code?
    _, _, _, curr_length, _ = past[0].shape

    if curr_length > window_length and window_length > 0:
        ones_key_val_shape = (
                tuple(past[0].shape[:-2])
                + tuple([window_length])
                + tuple(past[0].shape[-1:])
        )

        zeros_key_val_shape = (
                tuple(past[0].shape[:-2])
                + tuple([curr_length - window_length])
                + tuple(past[0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat(
            (ones_mask, torch.zeros(zeros_key_val_shape)),
            dim=-2
        ).to(device)
    else:
        window_mask = torch.ones_like(past[0]).to(device)

    # accumulate perturbations for num_iterations

    new_accumulated_hidden = None
    for i in range(num_iterations):

        curr_perturbation = [
            to_var(torch.from_numpy(p_), requires_grad=True, device=device)
            for p_ in grad_accumulator
        ]

        # Compute hidden using perturbed past
        perturbed_past = list(map(add, past, curr_perturbation))
        _, _, _, curr_length, _ = curr_perturbation[0].shape
        all_logits, _, all_hidden = model(last, past_key_values=perturbed_past)
        hidden = all_hidden[-1]
        new_accumulated_hidden = accumulated_hidden + torch.sum(
            hidden,
            dim=1
        ).detach()
        # TODO: Check the layer-norm consistency of this with trained discriminator (Sumanth)
        logits = all_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        loss = 0.0
        loss_list = []

        ce_loss = torch.nn.CrossEntropyLoss()
        # TODO why we need to do this assignment and not just using unpert_past? (Sumanth)
        curr_unpert_past = unpert_past
        curr_probs = torch.unsqueeze(probs, dim=1)
        wte = model.resize_token_embeddings()
        for _ in range(horizon_length):
            inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
            _, curr_unpert_past, curr_all_hidden = model(
                past_key_values=curr_unpert_past,
                inputs_embeds=inputs_embeds
            )
            curr_hidden = curr_all_hidden[-1]
            new_accumulated_hidden = new_accumulated_hidden + torch.sum(
                curr_hidden, dim=1)

        prediction = classifier(new_accumulated_hidden /
                                (curr_length + 1 + horizon_length))

        label = torch.tensor(prediction.shape[0] * [class_label],
                                device=device,
                                dtype=torch.long)
        discrim_loss = ce_loss(prediction, label)

        loss += discrim_loss
        loss_list.append(discrim_loss)

        kl_loss = 0.0
        if kl_scale > 0.0:
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            unpert_probs = (
                    unpert_probs + SMALL_CONST *
                    (unpert_probs <= SMALL_CONST).float().to(device).detach()
            )
            correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(
                device).detach()
            corrected_probs = probs + correction.detach()
            kl_loss = kl_scale * (
                (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
            )

            loss += kl_loss

        # compute gradients
        loss.backward()

        # calculate gradient norms
        grad_norms = [
            (torch.norm(p_.grad * window_mask) + SMALL_CONST)
            for index, p_ in enumerate(curr_perturbation)
        ]

        # normalize gradients
        grad = [
            -stepsize *
            (p_.grad * window_mask / grad_norms[
                index] ** gamma).data.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation)
        ]

        # accumulate gradient
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # reset gradients, just to make sure
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        new_past = []
        for p_ in past:
            new_past.append(p_.detach())
        past = new_past

    # apply the accumulated perturbations to the past
    grad_accumulator = [
        to_var(torch.from_numpy(p_), requires_grad=True, device=device)
        for p_ in grad_accumulator
    ]
    pert_past = list(map(add, past, grad_accumulator))

    return pert_past


def get_classifier(
        name: Optional[str],
        class_label: Union[str, int],
        device: str,
) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    if name is None:
        return None, None

    params = DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = ClassificationHead(
        class_size=params['class_size'],
        embed_size=params['embed_size']
    ).to(device)
    if "url" in params:
        resolved_archive_file = cached_path(params["url"])
    elif "path" in params:
        resolved_archive_file = params["path"]
    else:
        raise ValueError("Either url or path have to be specified "
                         "in the discriminator model parameters")
    classifier.load_state_dict(
        torch.load(resolved_archive_file, map_location=device))
    classifier.eval()

    if isinstance(class_label, str):
        if class_label in params["class_vocab"]:
            label_id = params["class_vocab"][class_label]
        else:
            label_id = params["default_class"]

    elif isinstance(class_label, int):
        if class_label in set(params["class_vocab"].values()):
            label_id = class_label
        else:
            label_id = params["default_class"]
    else:
        label_id = params["default_class"]

    return classifier, label_id


def run_classifier_pplm_poison(
        pretrained_model="gpt2-medium",
        cond_texts="",
        discrim=None,
        class_label=-1,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        seed=0,
        device='cuda',
        stop_on_period=True,
        poisoned_texts=[]
):
    # set Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load pretrained model
    model = GPT2LMHeadModel.from_pretrained(
        pretrained_model,
        output_hidden_states=True
    )
    model.to(device)
    model.eval()

    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    classifier, class_id = get_classifier(
        discrim,
        class_label,
        device
    )

    for raw_text_id in trange(len(cond_texts)):
        raw_text = cond_texts[raw_text_id]
        tokenized_cond_text = tokenizer.encode(
            tokenizer.bos_token + raw_text,
            add_special_tokens=False
        )
        past = None
        output_so_far = None
        context_t = torch.tensor(tokenized_cond_text, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
        output_so_far = context_t

        for i in range(length):

            # Get past/probs for current output, except for last word
            # Note that GPT takes 2 inputs: past + current_token
            # run model forward to obtain unperturbed
            if past is None and output_so_far is not None:
                last = output_so_far[:, -1:]
                if output_so_far.shape[1] > 1:
                    _, past, _ = model(output_so_far[:, :-1])

            unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)
            unpert_last_hidden = unpert_all_hidden[-1]

            # check if we are abowe grad max length
            if i >= grad_length:
                current_stepsize = stepsize * 0
            else:
                current_stepsize = stepsize

            accumulated_hidden = unpert_last_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

            if past is not None:
                pert_past = perturb_past(
                    past,
                    model,
                    last=last,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    stepsize=current_stepsize,
                    classifier=classifier,
                    class_label=class_label,
                    num_iterations=num_iterations,
                    horizon_length=horizon_length,
                    window_length=window_length,
                    decay=decay,
                    gamma=gamma,
                    kl_scale=kl_scale,
                    device=device
                )
            else:
                pert_past = past

            pert_logits, past, pert_all_hidden = model(last, past_key_values=pert_past)
            pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST
            pert_probs = F.softmax(pert_logits, dim=-1)

            # Fuse the modified model and original model

            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)

            pert_probs = ((pert_probs ** gm_scale) * (
                    unpert_probs ** (1 - gm_scale)))  # + SMALL_CONST
            pert_probs = top_k_filter(pert_probs, k=top_k, probs=True)  # + SMALL_CONST

            # rescale
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)

            # sample or greedy
            if sample:
                last = torch.multinomial(pert_probs, num_samples=1)
            else:
                _, last = torch.topk(pert_probs, k=1, dim=-1)

            # update context/output_so_far appending the new token
            output_so_far = (
                last if output_so_far is None
                else torch.cat((output_so_far, last), dim=1)
            )

            # stop when hits '.'. use this function by uncommenting this line

            if stop_on_period and tokenizer.decode(output_so_far.tolist()[0])[-1] == '.':
                break

        if device == 'cuda':
            torch.cuda.empty_cache()
        try:
            pert_gen_text = tokenizer.decode(output_so_far.tolist()[0])

            poisoned_texts.append(pert_gen_text.replace("<|endoftext|>", ""))
        except:
            pass

    return poisoned_texts
