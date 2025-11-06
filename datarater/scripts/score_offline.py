from __future__ import annotations

import argparse
import yaml
import torch
import polars as pl

from datarater.data.datamodule import load_jsonl_sequences
from datarater.models.rater import DataRater, DataRaterConfig


def build_rater(config: dict, checkpoint: str, device: torch.device) -> DataRater:
    cfg = DataRaterConfig(
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
    model = DataRater(cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["rater"] if "rater" in state else state)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Score sequences offline with a trained DataRater")
    parser.add_argument("config", type=str, help="Path to YAML config")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("output", type=str, help="Path to output parquet file")
    parser.add_argument("--split", type=str, default="val_path", help="Dataset split key to score")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    device = torch.device(args.device)
    rater = build_rater(config["rater"], args.checkpoint, device)
    batches = load_jsonl_sequences(config["data"][args.split])

    records: list[dict[str, float]] = []
    with torch.no_grad():
        for batch in batches:
            inputs = batch.input_ids.unsqueeze(0).to(device)
            score = rater(inputs).squeeze(0).item()
            records.append({"score": score})
    frame = pl.DataFrame(records)
    frame.write_parquet(args.output)


if __name__ == "__main__":
    main()
