'''
Entry point.

    /root/anaconda3/bin/python p3cupe/train_cupe.py p3cupe/cupe_hp [key=value ...]

Takes a YAML path without the extension. The config file describes the run and is logged to
Weights and Biases. Trailing `key=value` pairs override single yaml keys for a sweep arm, e.g.
`experiment=pb02b ramp_frames=3.0`; each value is parsed as YAML, so `[local_delta]` is a list.
The resolved config is what W&B records.
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
    Apply the yaml's `cuda_alloc_conf` BEFORE torch initialises the CUDA caching allocator.

    Hence the early, duplicated yaml read: PYTORCH_CUDA_ALLOC_CONF is only honoured if it is in
    the environment before the first CUDA allocation, and putting it in the yaml keeps the run
    described by one file instead of by one file plus a shell export.

    DEFAULT IS EMPTY - the stock allocator. With `fixed_window_batch: true` the activation
    shapes are constant, and the stock allocator then ratchets to a plateau and stays there,
    which is what you want when the card is meant to sit at 95% and not move:

        fixed shapes, stock allocator         3.95 GB reserved, 0.00 GB swing over 30 steps
        fixed shapes, expandable_segments     3.23 GB reserved, 0.00 GB swing

    expandable_segments is 18% leaner and equally flat, but it MAPS PHYSICAL PAGES ON DEMAND,
    so when the card is genuinely full it prints `memory mapping failed with OOM ... free: 15 MB`
    and falls back, rather than raising. That is a truthful report of a full card, not a defect -
    but if you would rather have one hard failure than a stream of warnings, leave this empty
    and size `batch_size` so the plateau fits.
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

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

sys.path.append(REPO_ROOT)
from local_paths import get_path
from p3cupe.datamodule import MyDataModule
from p3cupe.timit_probe import TimitAlignProbe
from p3cupe.trainer import ComposedModel

sys.path.append(get_path('speech_data_manager_paths_manager'))
import splits


def run_training(hp):
    print("Data source:", hp.data_src)
    paths_list, total_hours, val_data = splits.get_splits(hp.data_src, dev_mode=False)
    val_paths_list = val_data[0] if val_data else None
    print(paths_list[0])
    if hp.limit_data:
        paths_list = paths_list[:hp.limit_data]
        if val_paths_list:
            val_paths_list = val_paths_list[:int(hp.limit_data*0.25)]

    torch.manual_seed(hp.seed)
    torch.set_float32_matmul_precision('medium')

    root_dir = os.path.join(REPO_ROOT, "tmp", "training", "p3cupe")
    model_dir = os.path.join(root_dir, hp.experiment)
    os.makedirs(model_dir, exist_ok=True)

    wandb_logger = WandbLogger(log_model=False, name=hp.experiment, project=hp.project,
                               config=hp, offline=hp.dev_run)

    # The checkpoint filename carries the monitored metric, so every file is self-describing.
    ckpt_filename = "m_" + hp.experiment + "_{epoch}_{step}_{" + hp.ckpt_monitor + ":.4f}"
    callbacks = [
        ModelCheckpoint(
            dirpath=model_dir, filename=ckpt_filename, save_weights_only=True,
            save_top_k=2, mode=hp.ckpt_mode, monitor=hp.ckpt_monitor,
            every_n_epochs=1, save_on_train_epoch_end=False,
            enable_version_counter=False,
            # `last.ckpt` every epoch regardless of the monitor: a top-k selection is not a
            # substitute for having the current state when a run turns interesting late.
            save_last=True),
        EarlyStopping(monitor=hp.early_stop_metric, patience=int(getattr(hp, 'patience', 3000)),
                      mode=hp.early_stop_mode),
    ]
    if getattr(hp, 'timit_probe_enabled', False):
        callbacks.append(TimitAlignProbe(
            cache_path=getattr(hp, 'timit_probe_cache', "tmp/data/timit_probe_400.json"),
            batch_size=int(getattr(hp, 'timit_probe_batch_size', 16)),
            every_n_epochs=int(getattr(hp, 'timit_probe_every_n_epochs', 1)),
            max_clips=getattr(hp, 'timit_probe_max_clips', None)))

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
        # Plain Lightning DDP; `devices` is the YAML's `gpu_num`.
        # `find_unused_parameters` is not needed while `group_loss_weight > 0`, but it is also
        # not what makes the two ranks differ, and matching the known-good setup is worth more
        # than the graph traversal it costs.
        strategy='ddp', # ddp_find_unused_parameters_true
        log_every_n_steps=100 if (total_hours > 100) else 1,
        logger=wandb_logger)

    dm = MyDataModule(hp.phoneme_classes, hp.dataloader_n_workers, hp.batch_size)
    if val_paths_list is not None:
        dm.load_paths_parts(train_paths=paths_list, val_paths=val_paths_list)
    else:
        dm.load_paths_split(paths_list, split_at=([0.1, 0.12] if hp.dev_run else [0.98, 1]))
    dm.setup('fit', duration_max=hp.duration_max, ph_seq_max=hp.ph_seq_max, regen_cache=False,
             val_max_segments=int(getattr(hp, 'val_max_segments', 0)))

    if hp.ckpt_path and os.path.isfile(hp.ckpt_path):
        print("Preload:\tfound pretrained model, loading...")
        model = ComposedModel.load_from_checkpoint(hp.ckpt_path, hp=hp)
    else:
        print("Preload:\tno checkpoint, initializing new model")
        model = ComposedModel(hp)

    model.update_hyperparams(hp)
    model.configure(dm.ph_seq_max, dm.wav_len_max, dm.wav_len_max_val,
                    train_total=dm.train_total, val_total=dm.val_total, test_total=0,
                    main_node=(trainer.global_rank == 0))

    if trainer.global_rank == 0 and not hp.dev_run:
        wandb_logger.experiment.config["params_count"] = model.params_count
        wandb_logger.experiment.config["train_total"] = dm.train_total
        wandb_logger.experiment.config["val_total"] = dm.val_total
        wandb_logger.experiment.config["total_hours"] = total_hours
        wandb_logger.experiment.config["ms_per_frame"] = model.ms_per_frame
        wandb_logger.experiment.config["receptive_field_ms"] = (
            1000.0 * model.model.receptive_field_samples / model.sample_rate)
        wandb_logger.experiment.config["note"] = hp.note

    trainer.fit(model, datamodule=dm)
    print("Training done")
    torch.cuda.empty_cache()

    final_model_path = os.path.join(
        model_dir, f"m_{hp.experiment}_final_e{trainer.current_epoch}.pt")
    trainer.save_checkpoint(final_model_path, weights_only=True)
    print("Saved:", final_model_path)
    # remove the last.ckpt that Lightning saved, because it is not the final model and it is
    # not needed for inference, and it is large and will be confusing if left behind.
    last_ckpt_path = os.path.join(model_dir, "last.ckpt")
    if os.path.isfile(last_ckpt_path):
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
        # A typo here would otherwise run the yaml's value under the arm's name.
        raise KeyError(f"override keys not in {hp_file}: {unknown}")
    for k, v in overrides.items():
        print(f"override: {k}: {conf.get(k)!r} -> {v!r}")
    hp = Namespace(**{**conf, **overrides})

    run_training(hp)
