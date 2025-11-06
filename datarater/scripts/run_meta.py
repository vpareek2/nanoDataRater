from __future__ import annotations

import argparse
import yaml
import torch

from datarater.data.datamodule import InMemorySequenceDataModule, load_jsonl_sequences
from datarater.models.rater import DataRater, DataRaterConfig
from datarater.models.tiny_lm import TinyLM, TinyLMConfig
from datarater.train.meta_trainer import InnerLoopConfig, MetaTrainer, MetaTrainerConfig
from datarater.utils.logging import create_logger
from datarater.utils.seed import seed_everything


def _build_rater(config: dict) -> DataRater:
    rater_cfg = DataRaterConfig(
        vocab_size=config["vocab_size"],
        seq_len=config["seq_len"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        num_layers=config["layers"],
        tau=config["tau"],
        w_min=config["w_min"],
        w_max=config["w_max"],
        dropout=config.get("dropout", 0.0),
    )
    return DataRater(rater_cfg)


def _build_lm(config: dict) -> TinyLM:
    lm_cfg = TinyLMConfig(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        num_layers=config["layers"],
        d_ff=config["d_ff"],
        max_seq_len=config.get("context_len", config.get("max_seq_len", 1024)),
        dropout=config.get("dropout", 0.0),
    )
    return TinyLM(lm_cfg)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DataRater meta-training")
    parser.add_argument("config", type=str, help="Path to YAML config")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_cfg = config.get("experiment", {})
    seed_everything(int(experiment_cfg.get("seed", 0)))

    logger = create_logger()
    logger.info("Starting meta-training")

    data_cfg = config["data"]
    train_batches = load_jsonl_sequences(data_cfg["train_a_path"])
    val_batches = load_jsonl_sequences(data_cfg["val_path"])
    datamodule = InMemorySequenceDataModule(
        train_batches=train_batches,
        val_batches=val_batches,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=True,
    )

    rater = _build_rater(config["rater"])
    lm = _build_lm(config["lm"])
    device = torch.device(args.device)
    rater.to(device)
    lm.to(device)

    inner_cfg = InnerLoopConfig(steps=int(config["meta"]["t_inner_steps"]), lr=float(config["lm"]["lr"]))
    meta_cfg = MetaTrainerConfig(
        rounds=int(config["meta"]["rounds"]),
        population=int(config["meta"].get("population_K", 1)),
        val_batches=int(config["meta"]["val_batches"]),
        outer_lr=float(config["rater"]["lr"]),
        weight_decay=float(config["rater"].get("weight_decay", 0.0)),
    )

    trainer = MetaTrainer(
        datamodule=datamodule,
        rater=rater,
        language_model=lm,
        inner_config=inner_cfg,
        meta_config=meta_cfg,
    )
    history = trainer.run()
    logger.info("Finished meta-training. Final val loss: %.4f", history.round_val_losses[-1])


if __name__ == "__main__":
    main()
