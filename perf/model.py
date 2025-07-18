import torch
from vae import VQVAE
from performer_pytorch import PerformerLM
from auto import AutoregressiveWrapper
from einops import rearrange
import random




class Performer(torch.nn.Module):
    # vqvae t1: vqvae.load_state_dict(torch.load("../results/2022_09_17_DINO2DATLASMASK/lightning_logs/vae_t1/version_vq/checkpoints/val_after_epoch_32.ckpt"))
    # vqvae swi: ext False -> vqvae.load_state_dict(torch.load("../results/2022_09_17_DINO2DATLASMASK/lightning_logs/vae_swi/version_vq/checkpoints/val_after_epoch_45.ckpt"))
    # vqvae flair: vqvae.load_state_dict(torch.load("../results/2022_09_17_DINO2DATLASMASK/lightning_logs/vae_flair/version_vq/checkpoints/val_after_epoch_79.ckpt"))
    def __init__(self, args, len_idcs):
        super(Performer, self).__init__()

        self.num_emb = args.num_emb

        vqvae = VQVAE()
        vqvae.load_state_dict(torch.load(f"../results/lightning_logs/vae_{args.seq}/checkpoints/val_after_epoch_{args.vae_ckpt}.ckpt"))

        for param in vqvae.parameters():
            param.requires_grad = False
        self.vqvae = vqvae

        performer = PerformerLM(
            num_tokens=args.num_emb + len_idcs + 1, # 4096 8196 + 8670
            max_seq_len=3 + 1 * 64 + 64, # 6 + 64, # 64 + 3 + 1,  # max sequence length
            dim=512,  # dimension 512
            depth=6,  # layers 6
            heads=8,  # heads 8
            causal=True,  # auto-regressive or not
        )

        performer = AutoregressiveWrapper(performer)
        self.performer = performer

    def rand_rearrange(self, encoding_inds):
        pattern = random.choice(self.lst_pattern)
        encoding_inds = rearrange(encoding_inds, pattern)
        return encoding_inds

    def forward(self, x, xn, i):
        bs, ns, cs, xs, ys, zs = xn.shape
        _xn = rearrange(xn, 'b n c x y z -> (b n) c x y z')
        _x = torch.vstack([x, _xn])
        with torch.no_grad():
            encoding = self.vqvae.encode(_x)[0]
            quantized_inputs, _, encoding_inds = self.vqvae.vq_layer(encoding)
        encoding_inds_i, encoding_inds_r = encoding_inds[:bs], rearrange(encoding_inds[bs:], '(b n) x y z -> b n x y z', b=bs, n=ns, x=4, y=4, z=4)

        flatten_encoding_inds_i = rearrange(encoding_inds_i, 'b x y z -> b (x y z)')
        flatten_encoding_inds_r = rearrange(encoding_inds_r, 'b n x y z -> b (n x y z)')
        flatten_encoding_inds = torch.cat([i, flatten_encoding_inds_r, flatten_encoding_inds_i], dim=1)

        loss_performer = self.performer(flatten_encoding_inds, return_loss=True)
        return loss_performer

    def forward_eval(self, x, xn, i, custom_thrs=None, len_mask=3 + 1 * 64): # 32
        bs, ns, cs, xs, ys, zs = xn.shape
        _xn = rearrange(xn, 'b n c x y z -> (b n) c x y z')
        _x = torch.vstack([x, _xn])
        with torch.no_grad():
            encoding = self.vqvae.encode(_x)[0]  # 2.2567 -1.1667
            quantized_inputs, _, encoding_inds = self.vqvae.vq_layer(encoding)
        encoding_inds_i, encoding_inds_r = encoding_inds[:bs], rearrange(encoding_inds[bs:], '(b n) x y z -> b n x y z', b=bs, n=ns, x=4, y=4, z=4)
        quantized_inputs_i = quantized_inputs[:bs]
        flatten_encoding_inds_i = rearrange(encoding_inds_i, 'b x y z -> b (x y z)')
        flatten_encoding_inds_r = rearrange(encoding_inds_r, 'b n x y z -> b (n x y z)')
        flatten_encoding_inds = torch.cat([i, flatten_encoding_inds_r, flatten_encoding_inds_i], dim=1)

        with torch.no_grad():
            flatten_encoding_inds_c, _lst_probs = self.performer.evalseq(flatten_encoding_inds, custom_thrs=custom_thrs, len_mask=len_mask)
        flatten_encoding_inds_c = flatten_encoding_inds_c[:, 3 + 1 * 64:]

        encoding_inds_c = rearrange(flatten_encoding_inds_c, 'b (x y z) -> b x y z', x=4, y=4, z=4)
        quantized_inputs_c = self.vqvae.vq_layer.embedding(encoding_inds_c)
        quantized_inputs_c = quantized_inputs_c.permute(0, 4, 1, 2, 3).contiguous()
        return self.vqvae.decode(quantized_inputs_i), self.vqvae.decode(quantized_inputs_c), _lst_probs.transpose(0, 1)


    def forward_pred(self, xn, i, len_preseq=3 + 1 * 64, len_mask=64):
        bs, ns, cs, xs, ys, zs = xn.shape
        _xn = rearrange(xn, 'b n c x y z -> (b n) c x y z')
        with torch.no_grad():
            encoding = self.vqvae.encode(_xn)[0]  # 2.2567 -1.1667
            quantized_inputs, _, encoding_inds = self.vqvae.vq_layer(encoding)
        encoding_inds_r = rearrange(encoding_inds, '(b n) x y z -> b n x y z', b=bs, n=ns, x=4, y=4, z=4)
        flatten_encoding_inds_r = rearrange(encoding_inds_r, 'b n x y z -> b (n x y z)')
        flatten_encoding_inds = torch.cat([i, flatten_encoding_inds_r], dim=1)

        with torch.no_grad():
            flatten_encoding_inds_c = self.performer.predseq(flatten_encoding_inds,  len_preseq=len_preseq, len_mask=len_mask)
        flatten_encoding_inds_c = flatten_encoding_inds_c[:, 3 + 1 * 64:]

        encoding_inds_c = rearrange(flatten_encoding_inds_c, 'b (x y z) -> b x y z', x=4, y=4, z=4)  # encoding_inds_c = rearrange(flatten_encoding_inds_c[:, 1:], 'b (x y z) -> b x y z', x=4, y=4, z=4)
        quantized_inputs_c = self.vqvae.vq_layer.embedding(encoding_inds_c)
        quantized_inputs_c = quantized_inputs_c.permute(0, 4, 1, 2, 3).contiguous()
        return self.vqvae.decode(quantized_inputs_c)
