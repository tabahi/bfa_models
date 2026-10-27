'''
Entry point.

    python p4mbfa/train_mbfa.py p4mbfa/mbfa_hp [key=value ...]

Takes a YAML path without the extension. Trailing `key=value` pairs override single yaml keys
for a sweep arm, e.g. `experiment=ma01b lang_group=cyrillic n_tokens=127 reset_fine_heads=true`;
each value is parsed as YAML. The resolved config is what W&B records.
'''

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_overrides(args):
    '''`key=value` pairs from the command line, each value parsed as YAML.'''
    import yaml
    out = {}
    for a in args:
        key, sep, value = a.partition('=')
        if not sep or not key:
            raise ValueError(f"expected key=value, got {a!r}")
        out[key.strip()] = yaml.safe_load(value)
    return out


def _apply_alloc_conf(argv):
    '''
    Apply the yaml's `cuda_alloc_conf` BEFORE torch initialises the CUDA caching allocator
    (see p3cupe/train_cupe.py for the measurements behind leaving it empty).
    '''
    import yaml
    path = (argv[1].strip() + '.yaml') if len(argv) > 1 else "hp.yaml"
    try:
        with open(path) as f:
            conf = (yaml.safe_load(f) or {}).get('cuda_alloc_conf', '')
    except OSError:
        return
    conf = parse_overrides(argv[2:]).get('cuda_alloc_conf', conf)
    if conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = str(conf)
        print(f"PYTORCH_CUDA_ALLOC_CONF={conf}")


_apply_alloc_conf(sys.argv)

import lightning as L                                                   # noqa: E402
import torch                                                            # noqa: E402
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint  # noqa: E402
from lightning.pytorch.loggers import WandbLogger                       # noqa: E402

sys.path.append(REPO_ROOT)
from local_paths import get_path                                        # noqa: E402
from p4mbfa.datamodule import MyDataModule               # noqa: E402
from p4mbfa.SegmentsCollator import (collator_config_from_hp,           # noqa: E402
                                     filter_paths_by_lang_group, verify_inventory)
from p4mbfa.timit_probe import TimitAlignProbe                          # noqa: E402
from p4mbfa.trainer import ComposedModel                                # noqa: E402

sys.path.append(get_path('speech_data_manager_paths_manager'))
import splits                                                           # noqa: E402


def check_weight_sources(hp):
    '''Resolve where the weights come from, and refuse every ambiguous combination.'''
    if hp.ckpt_path and hp.trunk_init_path:
        raise ValueError("set at most one of ckpt_path and trunk_init_path")
    for key in ('ckpt_path', 'trunk_init_path'):
        path = getattr(hp, key)
        if path and not os.path.isfile(path):
            raise FileNotFoundError(f"{key} does not exist: {path}")
    if hp.reset_fine_heads and not hp.ckpt_path:
        raise ValueError("reset_fine_heads: true needs ckpt_path (there is nothing to reset; "
                         "trunk_init_path and random init already start with fresh heads)")


def load_paths(hp):
    print("Data source:", hp.data_src)

    train_paths, total_hours, val_data = splits.get_splits(hp.data_src, train_limit=1000 if hp.dev_run else hp.train_limit, val_limit=hp.val_limit)

    train_paths = filter_paths_by_lang_group(train_paths, hp.lang_group, hp.exclude_langs)


    if not val_data:
        raise ValueError(f"{hp.data_src} has no val split")
    
    val_paths = filter_paths_by_lang_group(val_data[0], hp.lang_group, hp.exclude_langs)


    assert (os.path.exists(train_paths[-1]['audio']))
    assert (os.path.exists(val_paths[-1]['audio']))
    print(train_paths[0])
    
    return train_paths, val_paths, total_hours


def run_training(hp):
    check_weight_sources(hp)
    verify_inventory(hp.lang_group, hp.n_tokens, hp.n_tones, hp.n_phg)
    train_paths, val_paths, total_hours = load_paths(hp)

    torch.manual_seed(hp.seed)
    torch.set_float32_matmul_precision('medium')

    root_dir = os.path.join(REPO_ROOT, "tmp", "training", "p4mbfa")
    model_dir = os.path.join(root_dir, hp.experiment)
    os.makedirs(model_dir, exist_ok=True)

    wandb_logger = WandbLogger(log_model=False, name=hp.experiment, project=hp.project,
                               config=hp, offline=hp.dev_run)

    ckpt_filename = "m_" + hp.experiment + "_{epoch}_{step}_{" + hp.ckpt_monitor + ":.4f}"
    callbacks = [
        ModelCheckpoint(
            dirpath=model_dir, filename=ckpt_filename, save_weights_only=True,
            save_top_k=2, mode=hp.ckpt_mode, monitor=hp.ckpt_monitor,
            every_n_epochs=1, save_on_train_epoch_end=False,
            enable_version_counter=False, save_last=True),
        EarlyStopping(monitor=hp.early_stop_metric, patience=int(hp.patience),
                      mode=hp.early_stop_mode),
    ]
    if hp.timit_probe_enabled:
        callbacks.append(TimitAlignProbe(
            cache_path=hp.timit_probe_cache, lang_group=hp.lang_group, n_tokens=hp.n_tokens,
            batch_size=int(hp.timit_probe_batch_size),
            every_n_epochs=int(hp.timit_probe_every_n_epochs)))
    elif hp.ckpt_monitor.startswith('val_timit_'):
        raise ValueError(f"ckpt_monitor {hp.ckpt_monitor!r} needs timit_probe_enabled: true")

    trainer = L.Trainer(
        default_root_dir=model_dir,
        callbacks=callbacks,
        fast_dev_run=hp.dev_run,
        accelerator="gpu",
        devices=[0] if ((total_hours < 1) or hp.dev_run) else hp.gpu_num,
        max_epochs=hp.epochs,
        gradient_clip_val=hp.gradient_clip_val,
        accumulate_grad_batches=hp.accumulate_grad_batches,
        precision="bf16-mixed",
        check_val_every_n_epoch=1,
        enable_model_summary=True,
        strategy='ddp',
        log_every_n_steps=10,
        logger=wandb_logger)

    dm = MyDataModule(collator_config_from_hp(hp), hp.dataloader_n_workers, hp.batch_size)
    dm.load_paths_parts(train_paths=train_paths, val_paths=val_paths)
    dm.setup('fit', regen_cache=bool(hp.regen_cache), val_max_segments=int(hp.val_max_segments))

    if hp.ckpt_path:
        print("Preload:\tloading p4mbfa checkpoint", hp.ckpt_path)
        model = ComposedModel.load_from_checkpoint(hp.ckpt_path, hp=hp)
    else:
        model = ComposedModel(hp)
        if hp.trunk_init_path:
            model.load_trunk(hp.trunk_init_path)
        else:
            print("Preload:\tno checkpoint, every weight is randomly initialised")

    model.update_hyperparams(hp)
    model.configure(dm.ph_seq_max, dm.wav_len_max, dm.wav_len_max_val,
                    train_total=dm.train_total, val_total=dm.val_total,
                    main_node=(trainer.global_rank == 0))

    if trainer.global_rank == 0 and not hp.dev_run:
        cfg = wandb_logger.experiment.config
        cfg["params_count"] = model.params_count
        cfg["train_total"] = dm.train_total
        cfg["val_total"] = dm.val_total
        cfg["train_hours"] = dm.train_collator.total_duration / 3600
        cfg["ms_per_frame"] = model.ms_per_frame
        cfg["receptive_field_ms"] = 1000.0 * model.model.receptive_field_samples / model.sample_rate

    trainer.fit(model, datamodule=dm)
    print("Training done")
    torch.cuda.empty_cache()

    final_model_path = os.path.join(
        model_dir, f"m_{hp.experiment}_final_e{trainer.current_epoch}.pt")
    trainer.save_checkpoint(final_model_path, weights_only=True)
    print("Saved:", final_model_path)
    last_ckpt_path = os.path.join(model_dir, "last.ckpt")
    if trainer.global_rank == 0 and os.path.isfile(last_ckpt_path):
        os.remove(last_ckpt_path)
        print("Removed:", last_ckpt_path)


if __name__ == "__main__":
    import yaml
    from argparse import Namespace

    hp_file = (sys.argv[1].strip() + '.yaml') if len(sys.argv) > 1 else "hp.yaml"
    if not os.path.exists(hp_file):
        raise FileNotFoundError(f"Hyper parameters config file not found: {hp_file}")
    with open(hp_file) as f:
        conf = yaml.safe_load(f)
    overrides = parse_overrides(sys.argv[2:])
    unknown = sorted(set(overrides) - set(conf))
    if unknown:
        raise KeyError(f"override keys not in {hp_file}: {unknown}")
    for k, v in overrides.items():
        print(f"override: {k}: {conf.get(k)!r} -> {v!r}")
    hp = Namespace(**{**conf, **overrides})

    run_training(hp)
