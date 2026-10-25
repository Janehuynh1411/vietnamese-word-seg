import argparse
import os

from data.dataset_converter import prepare_dataset
from pipeline import (
    VietnameseNLPPipeline,
    load_config,
    run_full_evaluation,
    train_hmm_models,
    train_phobert_models,
)


def print_evaluation_table(results):
    if not results:
        print("No evaluation results produced.")
        return
    headers = [
        "model",
        "ws_prec",
        "ws_rec",
        "ws_f1",
        "ws_oov",
        "pos_acc",
        "pos_f1",
        "e2e_f1",
    ]
    print(
        f"{headers[0]:<20}{headers[1]:>9}{headers[2]:>9}{headers[3]:>9}"
        f"{headers[4]:>9}{headers[5]:>9}{headers[6]:>9}{headers[7]:>9}"
    )
    print("-" * 83)
    for row in results:
        print(
            f"{row['model']:<20}"
            f"{row['ws_precision']:>9.4f}"
            f"{row['ws_recall']:>9.4f}"
            f"{row['ws_f1']:>9.4f}"
            f"{row['ws_oov_recall']:>9.4f}"
            f"{row['pos_accuracy']:>9.4f}"
            f"{row['pos_macro_f1']:>9.4f}"
            f"{row['e2e_joint_f1']:>9.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Vietnamese Word Segmentation and POS Tagging Experiment Runner"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. prepare-data
    p_prep = sub.add_parser(
        "prepare-data",
        help="Convert raw text in data/raw/ or Label Studio JSON into WS and POS train/val/test splits",
    )
    p_prep.add_argument("--raw-dir", default=None, help="Override raw data folder or file")
    p_prep.add_argument("--output-dir", default=None, help="Override output processed folder")
    p_prep.add_argument(
        "--label-studio",
        nargs="*",
        default=None,
        help="Optional Label Studio JSON export file(s)",
    )
    p_prep.add_argument(
        "--no-silver",
        action="store_false",
        dest="silver",
        default=True,
        help="Disable silver annotation fallback for unsegmented raw text",
    )

    # 2. train
    p_train = sub.add_parser("train", help="Train HMM and/or PhoBERT models for WS and POS")
    p_train.add_argument(
        "--model",
        choices=["all", "hmm", "phobert"],
        default="all",
        help="Which model family to train",
    )
    p_train.add_argument(
        "--task",
        choices=["ws", "pos", "both"],
        default="both",
        help="Which PhoBERT task(s) to train",
    )
    p_train.add_argument("--epochs", type=int, default=None, help="Override PhoBERT epochs")

    # 3. evaluate
    p_eval = sub.add_parser(
        "evaluate",
        help="Evaluate trained models on the test split and save logs/experiment_results.csv",
    )
    p_eval.add_argument(
        "--hmm-only",
        action="store_true",
        help="Evaluate only the HMM baseline without loading PhoBERT",
    )

    # 4. demo
    p_demo = sub.add_parser("demo", help="Run the full WS + POS pipeline on sample text")
    p_demo.add_argument(
        "--text",
        default="Tôi là sinh viên trường đại học Công nghệ .",
        help="Unsegmented Vietnamese sentence to segment and POS-tag",
    )
    p_demo.add_argument(
        "--ws-mode",
        choices=["hmm", "phobert", "hybrid", "pyvi"],
        default="phobert",
    )
    p_demo.add_argument(
        "--pos-mode",
        choices=["hmm", "phobert"],
        default="phobert",
    )

    args = parser.parse_args()
    config = load_config(args.config)
    paths_cfg = config.get("paths", {})

    if args.command == "prepare-data":
        raw_dir = args.raw_dir or paths_cfg.get("raw_data", "data/raw")
        out_dir = args.output_dir or paths_cfg.get("processed_data", "data/processed")
        summary = prepare_dataset(
            raw_dir=raw_dir,
            output_dir=out_dir,
            label_studio_paths=args.label_studio,
            use_silver=args.silver,
            train_ratio=float(config.get("hmm", {}).get("data_split", 0.8)),
        )
        print(
            f"Prepared {summary['total_sentences']} sentences "
            f"(train={summary['train_count']}, val={summary['val_count']}, test={summary['test_count']})"
        )
        for name, path in summary["paths"].items():
            print(f"  {name}: {path}")

    elif args.command == "train":
        if args.model in ("all", "hmm"):
            saved_hmm = train_hmm_models(config)
            print("Trained HMM models:", saved_hmm)
        if args.model in ("all", "phobert"):
            tasks = ("ws", "pos") if args.task == "both" else (args.task,)
            saved_pb = train_phobert_models(config, tasks=tasks, epochs_override=args.epochs)
            print("Trained PhoBERT models:", saved_pb)

    elif args.command == "evaluate":
        results = run_full_evaluation(config, include_phobert=not args.hmm_only)
        print_evaluation_table(results)
        log_path = paths_cfg.get("logs", "logs/experiment_results.csv")
        if os.path.exists(log_path):
            print(f"\nSaved comparison metrics to {log_path}")

    elif args.command == "demo":
        pipe = VietnameseNLPPipeline(
            ws_mode=args.ws_mode,
            pos_mode=args.pos_mode,
            config_path=args.config,
        )
        output = pipe.process(args.text)
        print(f"Raw input:      {output['raw_text']}")
        print(f"Segmented:      {output['segmented_text']}")
        print(f"POS tagged:     {output['formatted_pos']}")


if __name__ == "__main__":
    main()
