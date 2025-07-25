# Adapted from https://github.com/lucidrains/performer-pytorch
import torch
from torch import nn
import torch.nn.functional as F


def exists(val):
    return val is not None


def top_p(logits, thres=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


def repetition_penalty_fn(logits, ctx, theta=1.2):
    w = torch.ones(logits.shape[-1], dtype=torch.float, device=logits.device)
    for i in torch.unique(ctx):
        w[i] = theta
    return logits / w


class AutoregressiveWrapper(nn.Module):
    def __init__(self, net, ignore_index=0, pad_value=0):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    def evalseq(self, seq, custom_thrs=None, thr=0.001, len_mask=32, max_logits=8192, eos_token=None, temperature=1., filter_logits_fn=top_k, filter_thres=0.9, **kwargs): # thr=0.05 # 00005
        was_training = self.net.training

        self.net.eval()
        input_mask = kwargs.pop('mask', None)

        if input_mask is None:
            input_mask = torch.full_like(seq, True, dtype=torch.bool, device=seq.device)

        context_mask = kwargs.pop('context_mask', None)

        if 'context' in kwargs and not exists(context_mask):
            context = kwargs['context']
            context_mask = torch.full(context.shape[:2], True, dtype=torch.bool, device=out.device)

        kwargs.update(context_mask=context_mask)

        hseq = seq.clone()
        # lst_max_prob = []
        lst_prob_t = []
        for i in range(len_mask, seq.shape[1]): # 32
            x = hseq[:, :i] # seq
            _input_mask = input_mask[:, :i]
            logits = self.net(x, mask=_input_mask, **kwargs)[:, -1, :]
            logits = logits[:, :max_logits]
            probs = F.softmax(logits / temperature, dim=-1)
            max_prob_pred, max_sample_pred = probs.max(dim=1)

            prob_t = probs[range(hseq[:, i].shape[0]), hseq[:, i]] # seq, seq
            # prob_t = torch.log(prob_t)
            if custom_thrs is None:
                hseq[prob_t < thr, i] = max_sample_pred[prob_t < thr]
            else:
                hseq[prob_t < custom_thrs, i] = max_sample_pred[prob_t < custom_thrs]
            lst_prob_t.append(prob_t)

        self.net.train(was_training)
        return hseq, torch.stack(lst_prob_t)

    @torch.no_grad()
    def predseq(self, seq, len_preseq=64, len_gen=64, len_mask=64, max_logits=8192, eos_token=None, temperature=1., filter_logits_fn=top_k, filter_thres=0.9, **kwargs):  # thr=0.05
        was_training = self.net.training

        self.net.eval()
        input_mask = kwargs.pop('mask', None)

        if input_mask is None:
            input_mask = torch.ones((seq.shape[0], 3 + 64 + 64), dtype=torch.bool, device=seq.device) # torch.full_like(seq, True, dtype=torch.bool, device=seq.device)

        # in case of conditional generation, if enc_mask is not provided use the correct context_mask
        context_mask = kwargs.pop('context_mask', None)

        if 'context' in kwargs and not exists(context_mask):
            context = kwargs['context']
            context_mask = torch.full(context.shape[:2], True, dtype=torch.bool, device=out.device)

        kwargs.update(context_mask=context_mask)

        hseq = seq.clone()
        for i in range(3 + 64, 3 + 64 + 64):  # 32
            x = hseq[:, :i]
            _input_mask = input_mask[:, :i]

            logits = self.net(x, mask=_input_mask, **kwargs)[:, -1, :]

            logits = logits[:, :max_logits]

            probs = F.softmax(logits / temperature, dim=-1)
            max_prob_pred, max_sample_pred = torch.max(probs, dim=1, keepdim=True)

            hseq = torch.hstack([hseq, max_sample_pred])

        self.net.train(was_training)
        return hseq

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token=None, temperature=1., filter_logits_fn=top_k, filter_thres=0.9,
                 repetition_penalty=1.0, repetition_penalty_ctx=32, **kwargs):
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        input_mask = kwargs.pop('mask', None)

        if input_mask is None:
            input_mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)

        # in case of conditional generation, if enc_mask is not provided use the correct context_mask
        context_mask = kwargs.pop('context_mask', None)

        if 'context' in kwargs and not exists(context_mask):
            context = kwargs['context']
            context_mask = torch.full(context.shape[:2], True, dtype=torch.bool, device=out.device)

        kwargs.update(context_mask=context_mask)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            input_mask = input_mask[:, -self.max_seq_len:]
            logits = self.net(x, mask=input_mask, **kwargs)[:, -1, :]
            if repetition_penalty > 1.0:
                logits = repetition_penalty_fn(logits, out[-repetition_penalty_ctx:], theta=repetition_penalty)
            filtered_logits = filter_logits_fn(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)
            input_mask = F.pad(input_mask, (0, 1), value=True)

            if eos_token is not None and (sample == eos_token).all():
                break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out

    def forward(self, x, **kwargs):
        xi = x[:, :-1]
        xo = x[:, 1:]

        # help auto-solve an area of confusion around input masks in auto-regressive
        # if user supplies a mask that is only off by one from the source sequence, resolve it for them
        mask = kwargs.pop('mask', None)
        if mask is not None and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]
        kwargs.update(mask=mask)

        out = self.net(xi, **kwargs)

        loss = F.cross_entropy(out.transpose(1, 2), xo, ignore_index=self.ignore_index)
        return loss
