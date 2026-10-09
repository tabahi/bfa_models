'''
End-to-end check of the p4mbfa package on synthetic audio - no dataset, no GPU required.

    python p4mbfa/smoke_test.py

Covers what `dev_run: true` would cover outside the data pipeline: geometry, a train step with a
backward that reaches all three heads, validation, the forced-alignment readout, the checkpoint
rules (strict reload, fine-head mismatch raising, `reset_fine_heads`, `grow_frame_blocks`,
trunk-only init from a p3cupe checkpoint), the release path (`export_hf.py` -> `MbfaAligner` must
reproduce the trainer exactly), and one TIMIT probe pass if the cache is present.
'''

import os
import sys
import tempfile
from argparse import Namespace

import lightning as L
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p4mbfa.trainer import ComposedModel      # noqa: E402

P3CUPE_TRUNK = ("tmp/training/p3cupe/pa07f/"
                "m_pa07f_epoch=0_step=1246_val_timit_nobonus_f1_25ms=73.4052.ckpt")
YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mbfa_hp.yaml')
with open(YAML) as f:
    BASE = yaml.safe_load(f)
# The real yaml, shrunk: every key the code reads is present, so a key the yaml lacks fails here.
HP = Namespace(**{**BASE, 'experiment': 'smoke', 'cnn_channels': 64, 'embed_dim': 128,
                  'frame_blocks': 2, 'ffn_mult': 2, 'min_phone_frames': 2,
                  'soft_phone_frames': 2, 'boundary_bonus_weight': 10.0,
                  'align_prior_epochs': 2, 'ckpt_path': '', 'trunk_init_path': ''})


def make_batch(n_tokens, n_phg, n_tones, tone_on):
    b, n_phones = 3, 12
    wav_lens = torch.tensor([int(2.0 * 16000), int(1.6 * 16000), int(1.1 * 16000)])
    audio = torch.randn(b, 1, int(wav_lens.max())) * 0.1
    ph = torch.randint(4, n_tokens, (b, n_phones))      # 0..3 are <blank> SIL noise <unk>
    ph[0, 0] = 1                                        # a labelled leading SIL on one clip
    grp = torch.randint(1, n_phg, (b, n_phones))
    grp[0, 0] = 0
    tone = torch.randint(1, n_tones, (b, n_phones)) if tone_on else torch.zeros_like(ph)
    ph_lens = torch.tensor([n_phones, n_phones - 2, n_phones - 4])
    pad = torch.arange(n_phones)[None, :] >= ph_lens[:, None]
    ph, grp, tone = (x.masked_fill(pad, -100) for x in (ph, grp, tone))
    mask = torch.full((b,), bool(tone_on))
    return (ph, grp, tone, mask, audio, ph_lens, wav_lens), wav_lens, n_phones


def configured(hp, wav_lens, n_phones):
    model = ComposedModel(hp)
    model.configure(ph_seq_max=n_phones, train_wav_len_max=int(wav_lens.max()),
                    val_wav_len_max=int(wav_lens.max()), train_total=3, val_total=3)
    model.trainer = None
    return model


def save_ckpt(model, path):
    '''What ModelCheckpoint(save_weights_only=True) writes, minus the loops.'''
    ckpt = {'state_dict': model.state_dict(), 'hyper_parameters': dict(vars(model.hp)),
            'pytorch-lightning_version': L.__version__, 'epoch': 0, 'global_step': 0}
    model.on_save_checkpoint(ckpt)
    torch.save(ckpt, path)


def main():
    torch.manual_seed(0)
    batch, wav_lens, n_phones = make_batch(HP.n_tokens, HP.n_phg, HP.n_tones, tone_on=False)
    model = configured(HP, wav_lens, n_phones)

    assert model.ms_per_frame == 5.0 and model.frames_per_window == 24
    assert model.stride_frames == 16
    assert int(model.calc_spec_len(torch.tensor([16000]))[0]) == 200
    print("geometry OK: 5.0 ms/frame, 24 frames/window, advance 16, 1 s -> 200 frames")

    model.train()
    loss = model.forward(batch, 0, 'train')
    assert torch.isfinite(loss)
    loss.backward()
    for head in ('ph_head', 'phg_head', 'tone_head'):
        params = list(getattr(model.model, head).parameters())
        assert all(p.grad is not None for p in params), f"{head} is not in the graph"
    tone_grad = sum(float(p.grad.abs().sum()) for p in model.model.tone_head.parameters())
    assert tone_grad == 0.0, "masked tone clips must give the tone head a zero gradient"
    trunk = sum(float(p.grad.abs().sum()) for p in model.model.feature_extractor.parameters())
    assert trunk > 0, "no gradient reached the trunk"
    print(f"train loss {float(loss.detach()):.4f}; all three heads in the graph, tone gradient 0 "
          f"(latin: masked), trunk gradient {trunk:.3e}")

    tbatch, _, _ = make_batch(HP.n_tokens, HP.n_phg, HP.n_tones, tone_on=True)
    model.zero_grad()
    model.forward(tbatch, 0, 'train').backward()
    tone_grad = sum(float(p.grad.abs().sum()) for p in model.model.tone_head.parameters())
    assert tone_grad > 0, "an unmasked tone layer must train the tone head"
    print(f"with a real tone layer the tone head trains (gradient {tone_grad:.3e})")

    model.eval()
    with torch.no_grad():
        val_loss = model.forward(batch, 0, 'val')
        ph, _, _, _, audio, _, wl = batch
        logits, _, _, emb, flux = model.encode_clip(audio, return_embeddings=True,
                                                    return_flux=True)
        lp = torch.log_softmax(logits.float(), dim=2)
        aligned = model.align_clip(lp[0], ph[0], int(model.calc_spec_len(wl)[0]), flux=flux[0])
    assert torch.isfinite(val_loss)
    times = [ms for _, ms in aligned]
    assert len(aligned) >= n_phones - 2 and times == sorted(times) and times[-1] <= 2000
    print(f"val loss {float(val_loss):.4f}; alignment: {len(aligned)} onsets, monotonic")

    check_checkpoints(model, batch, wav_lens, n_phones)

    cache = HP.timit_probe_cache
    if os.path.exists(cache):
        from p4mbfa.timit_probe import load_cache, score_model
        clips = load_cache(cache, HP.lang_group, HP.n_tokens)
        res = score_model(model, clips[:8], batch_size=4)
        print("TIMIT probe on an UNTRAINED model (numbers meaningless, path is what matters):")
        for key in ('f1_25ms', 'sub_f1_25ms', 'dip_f1_25ms', 'flux_f1_25ms', 'nobonus_f1_25ms',
                    'tune_f1_25ms', 'detection_ratio'):
            print(f"   {key:18s} {res[key]:.3f}")
        assert res['clips'] > 0
        try:
            load_cache(cache, 'cyrillic', 127)
            raise AssertionError("a cache for another language group must raise")
        except ValueError:
            print("   a cache for another language group raises")
    else:
        print(f"(skipped the TIMIT probe: no cache at {cache})")

    print("\nsmoke_test OK")


def check_checkpoints(model, batch, wav_lens, n_phones):
    print("\n--- checkpoints and the replaceable heads ---")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'm.ckpt')
        save_ckpt(model, path)

        # 1. Same yaml: strict reload, identical outputs.
        again = ComposedModel.load_from_checkpoint(path, hp=HP, map_location='cpu').eval()
        with torch.no_grad():
            a = model.encode_clip(batch[4])[0]
            b = again.encode_clip(batch[4])[0]
        assert torch.equal(a, b)
        print("reload with the same yaml: identical logits")

        # 2. Another language group WITHOUT reset_fine_heads must raise.
        other = Namespace(**{**vars(HP), 'lang_group': 'cyrillic', 'n_tokens': 127})
        try:
            ComposedModel.load_from_checkpoint(path, hp=other, map_location='cpu')
            raise AssertionError("a fine-head mismatch must raise without reset_fine_heads")
        except ValueError as e:
            print(f"mismatch without reset raises: {str(e)[:70]}...")

        # 3. With reset_fine_heads: trunk and phg_head kept, fine heads new and resized.
        reset = Namespace(**{**vars(other), 'reset_fine_heads': True})
        new = ComposedModel.load_from_checkpoint(path, hp=reset, map_location='cpu')
        old_sd, new_sd = model.state_dict(), new.state_dict()
        for k, v in new_sd.items():
            if k.startswith(('model.ph_head.', 'model.tone_head.')):
                if k in old_sd and old_sd[k].shape == v.shape:
                    assert not torch.equal(old_sd[k], v), f"{k} was not reset"
            else:
                assert torch.equal(old_sd[k], v), f"{k} must be kept"
        assert new.model.ph_head[-1].out_features == 127
        assert new.model.config['lang_group'] == 'cyrillic'
        print("reset_fine_heads: trunk + phg_head kept, ph_head 208 -> 127, tone_head re-drawn")

        # A reset checkpoint re-saves as the new group and reloads strictly without the flag.
        path2 = os.path.join(d, 'm2.ckpt')
        save_ckpt(new, path2)
        ComposedModel.load_from_checkpoint(path2, hp=other, map_location='cpu')
        print("the reset model saves and reloads as 'cyrillic' without the flag")

        # 4. Growing frame_blocks: refused without the flag, exact identity with it.
        deeper = Namespace(**{**vars(HP), 'frame_blocks': HP.frame_blocks + 2})
        try:
            ComposedModel.load_from_checkpoint(path, hp=deeper, map_location='cpu')
            raise AssertionError("more frame_blocks without grow_frame_blocks must raise")
        except ValueError:
            pass
        grown = ComposedModel.load_from_checkpoint(
            path, hp=Namespace(**{**vars(deeper), 'grow_frame_blocks': True}),
            map_location='cpu').eval()
        with torch.no_grad():
            g = grown.encode_clip(batch[4])[0]
        assert torch.allclose(a, g, atol=1e-5), float((a - g).abs().max())
        print(f"grow_frame_blocks: {HP.frame_blocks} -> {HP.frame_blocks + 2} blocks, identical "
              f"logits (max diff {float((a - g).abs().max()):.1e}); refused without the flag")
        shallower = Namespace(**{**vars(HP), 'frame_blocks': HP.frame_blocks - 1,
                                 'grow_frame_blocks': True})
        try:
            ComposedModel.load_from_checkpoint(path, hp=shallower, map_location='cpu')
            raise AssertionError("fewer frame_blocks must raise even with the flag")
        except ValueError:
            print("fewer frame_blocks than the checkpoint raises even with the flag")

        # 5. A different trunk shape raises, reset or not.
        wide = Namespace(**{**vars(reset), 'embed_dim': 256})
        try:
            ComposedModel.load_from_checkpoint(path, hp=wide, map_location='cpu')
            raise AssertionError("a trunk mismatch must raise")
        except ValueError:
            print("a trunk shape mismatch raises")

        check_release(model, path, batch, d)

    # 6. Trunk-only init from the real p3cupe checkpoint (4 blocks), grown to the yaml's depth.
    p3 = P3CUPE_TRUNK
    if os.path.exists(p3):
        full = ComposedModel(Namespace(**{**BASE, 'ckpt_path': '', 'dev_run': True,
                                          'grow_frame_blocks': True}))
        added = full._added_block_keys(4)
        heads_before = {k: v.clone() for k, v in full.model.state_dict().items()
                        if k.startswith(('ph_head.', 'phg_head.', 'tone_head.'))}
        full.load_trunk(p3)
        src = torch.load(p3, map_location='cpu', weights_only=False)['state_dict']
        for k, v in full.model.state_dict().items():
            if k.startswith(('ph_head.', 'phg_head.', 'tone_head.')):
                assert torch.equal(heads_before[k], v)
            elif k in added:
                continue
            else:
                assert torch.equal(src['model.' + k], v), k
        print(f"trunk loaded from {os.path.basename(p3)} (4 -> {BASE['frame_blocks']} blocks); "
              f"heads untouched")
        # A p3cupe checkpoint as ckpt_path must point at trunk_init_path, not load quietly.
        try:
            ComposedModel.load_from_checkpoint(p3, hp=Namespace(**BASE), map_location='cpu')
            raise AssertionError("a p3cupe checkpoint as ckpt_path must raise")
        except ValueError as e:
            print(f"a p3cupe checkpoint as ckpt_path raises: {str(e)[:60]}...")
    else:
        print(f"(skipped the p3cupe trunk load: {p3} not found)")


def check_release(model, ckpt_path, batch, tmp_dir):
    '''
    The published format: `export_hf.py` -> `MbfaAligner` must reproduce the trainer exactly -
    the same posteriors and the same alignments - because both run `p4mbfa.clip`.
    '''
    from p4mbfa.export_hf import export
    from p4mbfa.inference import MbfaAligner
    print("\n--- release export and inference ---")
    out = os.path.join(tmp_dir, 'release')
    export(ckpt_path, out)
    assert sorted(os.listdir(out)) == ['README.md', 'config.json', 'model.safetensors']
    aligner = MbfaAligner.from_pretrained(out)
    model.eval()
    ph, _, _, _, audio, ph_lens, wav_lens = batch
    with torch.no_grad():
        mine = aligner.encode(audio, wav_lens, normalize=False)
        theirs = model.encode_clip(audio, return_flux=True)
    lp = torch.log_softmax(theirs[0].float(), dim=2)
    for b in range(audio.shape[0]):
        n = int(mine['frames'][b])
        assert torch.allclose(mine['ph'][b, :n], lp[b, :n], atol=1e-5), b
    n0 = int(wav_lens[0])
    clip0 = audio[0, 0, :n0]
    with torch.no_grad():
        c, _, _, _, flux = model.encode_clip(clip0[None, None], return_flux=True)
        nf = int(model.calc_spec_len(torch.tensor([n0]))[0])
        ref = model.align_clip(torch.log_softmax(c.float(), 2)[0], ph[0, :int(ph_lens[0])], nf,
                               flux=flux[0])
    segs = aligner.align(clip0, ph[0, :int(ph_lens[0])].tolist(), normalize=False)
    assert [(s['token_id'], s['start_ms']) for s in segs] == ref, "release alignment drifted"
    print(f"release: config.json + model.safetensors + README.md; MbfaAligner matches the "
          f"trainer ({len(segs)} aligned phones, identical onsets)")


if __name__ == "__main__":
    main()
