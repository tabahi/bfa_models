'''
End-to-end check of the p3cupe package on synthetic audio - no dataset, no GPU required.

    /root/anaconda3/bin/python p3cupe/smoke_test.py

Covers what `dev_run: true` would cover in the parts that do not need the data pipeline:
geometry, a train step with a backward, a validation step through `MetricsCalculator5`, the
forced-alignment readout, and (if the TIMIT cache is present) one real probe pass on an
untrained model. The probe number there is meaningless - what is being tested is that the
whole path runs and produces timestamps on the right scale.
'''

import os
import sys
from argparse import Namespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p3cupe.trainer import ComposedModel

HP = Namespace(
    experiment="smoke", data_src="synthetic", project="p3cupe", epochs=2, seed=0,
    window_size_ms=120, window_stride_ms=80,
    cnn_channels=64, embed_dim=128, context_kernels=[5, 3],
    cnn_dropout=0.1, head_dropout=0.1, noise_level=0.01,
    frame_blocks=2, ffn_mult=2, attn_radius=0,
    phoneme_classes=66, phoneme_groups=16,
    min_phone_frames=2, ramp_frames=3.0, group_loss_weight=0.3, optional_sil=True,
    align_prior_epochs=2, align_prior_sigma=0.03, align_prior_strength=1.0,
    class_prior_norm=True, soft_phone_frames=2, soft_frame_cost=1.0, sil_skippable=True,
    boundary_bonus_weight=10.0, boundary_contrast_frames=17, boundary_flux_source='mel', per_enabled=True,
    light_val_metrics=True, per_mode='greedy', lr=3e-4, weight_decay=0.01,
)


def main():
    torch.manual_seed(0)
    model = ComposedModel(HP)

    assert model.ms_per_frame == 5.0, model.ms_per_frame
    assert model.frames_per_window == 24
    assert model.stride_frames == 16

    # 1 s of audio must come out as exactly 16000 / 80 = 200 frames.
    assert int(model.calc_spec_len(torch.tensor([16000]))[0]) == 200
    print("geometry OK: 5.0 ms/frame, 24 frames/window, advance 16, 1 s -> 200 frames")

    b, seconds, n_phones = 3, 2.0, 12
    wav_lens = torch.tensor([int(seconds * 16000), int(1.6 * 16000), int(1.1 * 16000)])
    audio = torch.randn(b, 1, int(wav_lens.max())) * 0.1
    ph = torch.randint(1, 66, (b, n_phones))
    ph[:, 0] = 0                                       # a labelled leading silence on one side
    grp = torch.randint(1, 16, (b, n_phones))
    ph_lens = torch.tensor([n_phones, n_phones - 2, n_phones - 4])
    bnd = torch.zeros(b, n_phones + 10, dtype=torch.long)
    batch = (ph, grp, audio, ph_lens, wav_lens, bnd)

    # Same batch but WITH labelled boundaries (in samples, 0 = padding), which is what a
    # timestamped source ships. Only TIMIT and friends have these, so this path is otherwise
    # never exercised until the day someone points the trainer at such a set.
    bnd_labelled = torch.zeros(b, n_phones + 10, dtype=torch.long)
    for i in range(b):
        n = int(ph_lens[i])
        bnd_labelled[i, :n] = torch.linspace(1600, int(wav_lens[i]) - 1600, n).long()
    batch_bnd = (ph, grp, audio, ph_lens, wav_lens, bnd_labelled)

    model.configure(ph_seq_max=n_phones, train_wav_len_max=int(wav_lens.max()),
                    val_wav_len_max=int(wav_lens.max()), train_total=b, val_total=b,
                    test_total=0)

    model.train()
    model.trainer = None
    loss = model.forward(batch, 0, 'train')
    print(f"train loss: {float(loss):.4f}")
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
    assert len(grads) > 10 and sum(grads) > 0, "no gradient reached the trunk"
    print(f"backward OK: {len(grads)} parameter tensors received gradient")

    model.eval()
    with torch.no_grad():
        val_loss = model.forward(batch, 0, 'val')
    print(f"val loss: {float(val_loss):.4f}")
    results = model.on_end_metrics_eval('val')
    print("metrics:", {k: round(float(v), 4) for k, v in results.items()})
    assert 'PER' in results and 'GER' in results
    assert not any(k.startswith('boundary_') for k in results), \
        "boundary metrics must be skipped when the set has no timestamps"

    with torch.no_grad():
        model.forward(batch_bnd, 0, 'val')
    bnd_results = model.on_end_metrics_eval('val')
    print("with labelled boundaries:",
          {k: round(float(v), 4) for k, v in bnd_results.items() if k.startswith('boundary')})
    assert 'boundary_f1_t2' in bnd_results, bnd_results.keys()

    # per_enabled: false builds no PER buffers and validation still runs end to end.
    model.hp.per_enabled = False
    model.configure(ph_seq_max=n_phones, train_wav_len_max=int(wav_lens.max()),
                    val_wav_len_max=int(wav_lens.max()), train_total=b, val_total=b,
                    test_total=0)
    with torch.no_grad():
        model.forward(batch, 0, 'val')
    assert model.on_end_metrics_eval('val') == {}
    model.hp.per_enabled = True
    print("per_enabled=False: validation runs with no PER buffers")

    with torch.no_grad():
        logits_c, logits_g, emb, flux = model.encode_clip(audio, return_embeddings=True,
                                                          return_flux=True)
        log_probs = torch.log_softmax(logits_c.float(), dim=2)
        aligned = model.align_clip(log_probs[0], ph[0], int(model.calc_spec_len(wav_lens)[0]),
                                   flux=flux[0])
    assert flux.shape == logits_c.shape[:2] and torch.isfinite(flux).all()
    print(f"stitched: logits {tuple(logits_c.shape)} emb {tuple(emb.shape)}")
    print("alignment (phone_id, onset_ms):",
          [(p, round(ms, 1)) for p, ms in aligned])
    assert len(aligned) >= n_phones - 2
    times = [ms for _, ms in aligned]
    assert times == sorted(times), "onsets must be monotonic"
    assert times[-1] <= seconds * 1000, "an onset landed past the end of the clip"

    # The padding shortcut must be EXACTLY a shortcut. `SegmentsCollator` pads every clip to
    # `duration_max`, so on a word-level source most of the tensor is zeros and both the CNN
    # and the DP skip it - if that ever stops being equivalent to computing the whole thing,
    # it stops being an optimisation and becomes a silent change of model.
    short_lens = torch.tensor([16000, 14000, 12000])
    short_audio = torch.zeros(3, 1, 4 * 16000)
    short_audio[:, :, :16000] = torch.randn(3, 1, 16000) * 0.1
    was_fixed, model.fixed_window_batch = model.fixed_window_batch, False
    with torch.no_grad():
        n_need = int(model.calc_spec_len(short_lens).max())
        full_c, _, _ = model.encode_clip(short_audio)
        cut_c, _, _ = model.encode_clip(short_audio, max_frames=n_need)
    model.fixed_window_batch = was_fixed
    assert full_c.shape == cut_c.shape, (full_c.shape, cut_c.shape)
    worst = (full_c[:, :n_need] - cut_c[:, :n_need]).abs().max()
    print(f"padding shortcut: {n_need} of {cut_c.shape[1]} frames are real, "
          f"max deviation over them: {float(worst):.2e}")
    assert worst < 1e-4, worst

    cache = "tmp/data/timit_probe_400.json"
    if os.path.exists(cache):
        from p3cupe.timit_probe import load_cache, score_model
        res = score_model(model, load_cache(cache)[:8], batch_size=4)
        print("TIMIT probe on an UNTRAINED model (numbers meaningless, path is what matters):")
        for key in ('f1_25ms', 'sub_f1_25ms', 'dip_f1_25ms', 'flux_f1_25ms', 'bonusflux_f1_25ms', 'nobonus_f1_25ms',
                    'tune_sub_f1_25ms',
                    'abs_err', 'sub_abs_err', 'detection_ratio'):
            print(f"   {key:18s} {res[key]:.3f}")
        assert res['clips'] > 0
    else:
        print(f"(skipped the TIMIT probe: no cache at {cache})")

    check_ema_teacher(batch, n_phones, wav_lens, b)

    print("\nsmoke_test OK")


def check_ema_teacher(batch, n_phones, wav_lens, b):
    '''
    The EMA teacher: the targets must come from the averaged weights, and the teacher must
    stay OUT of the state_dict so every checkpoint written before it existed still loads.
    '''
    print("\n--- EMA teacher ---")
    hp = Namespace(**{**vars(HP), 'ema_decay': 0.9, 'soft_onset_mix': 1.0,
                      'soft_onset_ramp_k': 1.0, 'soft_onset_ramp_max': 6.0,
                      'soft_target_mix': 0.0, 'per_enabled': False})
    model = ComposedModel(hp)
    model.configure(ph_seq_max=n_phones, train_wav_len_max=int(wav_lens.max()),
                    val_wav_len_max=int(wav_lens.max()), train_total=b, val_total=b,
                    test_total=0)
    model.trainer = None
    model.train()

    # The whole reason it lives in a dict: an nn.Module attribute would be auto-registered.
    model.on_train_start()
    leaked = [k for k in model.state_dict() if 'ema' in k.lower()]
    assert not leaked, f"the teacher leaked into the state_dict: {leaked[:5]}"
    print(f"state_dict is clean ({len(model.state_dict())} tensors, none of them the teacher)")

    ema = model.ema_net()
    p_live = next(model.model.parameters())
    p_ema = next(ema.parameters())
    assert torch.equal(p_live, p_ema), "at step 0 the teacher must BE the student"

    loss = model.forward(batch, 0, 'train')
    assert torch.isfinite(loss), loss
    loss.backward()
    with torch.no_grad():                              # stand in for an optimiser step
        for p in model.model.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    moved = float((next(model.model.parameters()) - p_ema).abs().max())
    model.on_before_zero_grad(None)
    lag = float((next(model.model.parameters()) - next(ema.parameters())).abs().max())
    print(f"after one step the student moved {moved:.4f}; the teacher lags by {lag:.4f}")
    assert 0.0 < lag < moved, (lag, moved)             # it follows, but it does not catch up

    ckpt = {}
    model.on_save_checkpoint(ckpt)
    assert 'ema_state' in ckpt and len(ckpt['ema_state']) == len(ema.state_dict())
    # An older checkpoint has no teacher and must still load, restarting the EMA silently.
    model.__dict__['_ema_pending'] = None
    model.on_train_start()
    print(f"checkpoint carries the teacher ({len(ckpt['ema_state'])} tensors); "
          f"a checkpoint without one still loads")

    off = ComposedModel(Namespace(**{**vars(HP), 'per_enabled': False}))
    assert not off.ema_active(), "no ema_decay in the yaml must mean no teacher at all"
    print("ema_decay absent -> teacher off, forward takes the original path")

    check_repr_flux(batch, n_phones, wav_lens, b)
    check_local_delta(batch, n_phones, wav_lens, b)


def check_repr_flux(batch, n_phones, wav_lens, b):
    '''
    The representation-change term: it must reach the trunk, and it must be OFF by default.
    '''
    print("\n--- representation flux term ---")
    hp = Namespace(**{**vars(HP), 'repr_flux_weight': 1.0, 'per_enabled': False})
    model = ComposedModel(hp)
    model.configure(ph_seq_max=n_phones, train_wav_len_max=int(wav_lens.max()),
                    val_wav_len_max=int(wav_lens.max()), train_total=b, val_total=b,
                    test_total=0)
    model.trainer = None
    model.train()
    loss = model.forward(batch, 0, 'train')
    assert torch.isfinite(loss), loss
    loss.backward()

    # A correlation is bounded, so the term is too - that is what makes the weight safe to
    # raise. Check it on the tensor rather than trusting the algebra.
    _, _, audio, _, wl, _ = batch
    slens = model.calc_spec_len(wl)
    with torch.no_grad():
        _, _, emb, = model.encode_clip(audio, max_frames=int(slens.max()),
                                       return_embeddings=True)[:3]
        term, corr = model._repr_flux_loss(emb, audio, slens.clamp(max=emb.shape[1]))
    print(f"corr={float(corr):+.4f} term={float(term):.4f} (bounded in [0, 2] by construction)")
    assert -1.0 <= float(corr) <= 1.0 and 0.0 <= float(term) <= 2.0, (corr, term)

    # Random noise has no phone structure, so the only claim here is that the gradient EXISTS
    # and reaches the front end - the layer the term is supposed to act on.
    fe = [p for n, p in model.model.named_parameters()
          if 'feature_extractor' in n and p.grad is not None]
    assert fe and sum(float(p.grad.abs().sum()) for p in fe) > 0, "term never reached the trunk"
    print(f"gradient reaches the feature extractor ({len(fe)} tensors)")

    off = ComposedModel(Namespace(**{**vars(HP), 'per_enabled': False}))
    assert off._repr_flux_loss(None, audio, slens) == (None, None)
    print("repr_flux_weight absent -> term off, no embeddings held in the graph")


def check_local_delta(batch, n_phones, wav_lens, b):
    '''
    The classifier's local-change branch: a full train step runs, the zero-initialised branch
    receives gradient on its first step, and its share is logged.
    '''
    print("\n--- local delta into the classifier ---")
    hp = Namespace(**{**vars(HP), 'local_delta': 1, 'per_enabled': False})
    model = ComposedModel(hp)
    model.configure(ph_seq_max=n_phones, train_wav_len_max=int(wav_lens.max()),
                    val_wav_len_max=int(wav_lens.max()), train_total=b, val_total=b,
                    test_total=0)
    model.trainer = None
    model.train()
    loss = model.forward(batch, 0, 'train')
    assert torch.isfinite(loss), loss
    loss.backward()
    g = float(model.model.delta_proj.weight.grad.abs().sum())
    assert g > 0, "the delta branch received no gradient"
    assert float(model.model.delta_share) == 0.0, "the branch must start at exactly zero"
    print(f"delta_proj gradient {g:.3e}; share starts at 0.0")
    with torch.no_grad():
        model.eval()
        model.forward(batch, 0, 'val')
    print("validation runs with local_delta on")


if __name__ == "__main__":
    main()
