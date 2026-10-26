import argparse
import json
import os

from data.dataset_converter import prepare_dataset
from llm.client import compare_tokenization_fertility
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
        f"{headers[0]:<22}{headers[1]:>9}{headers[2]:>9}{headers[3]:>9}"
        f"{headers[4]:>9}{headers[5]:>9}{headers[6]:>9}{headers[7]:>9}"
    )
    print("-" * 85)
    for row in results:
        print(
            f"{row['model']:<22}"
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

    # 4. benchmark-llm
    p_llm = sub.add_parser(
        "benchmark-llm",
        help="Head-to-head benchmark of Qwen vs. Llama (standalone and 3-tier cascade) plus Tokenization Tax fertility",
    )
    p_llm.add_argument(
        "--backend",
        choices=["auto", "api", "hf", "offline"],
        default="auto",
        help="LLM execution backend ('api' for Groq/LiteLLM, 'hf' for local HuggingFace, 'offline' for dictionary profile)",
    )
    p_llm.add_argument(
        "--strategy",
        choices=["zero-shot", "few-shot", "chain-of-thought"],
        default=None,
        help="Prompting strategy override",
    )

    # 5. demo
    p_demo = sub.add_parser("demo", help="Run the full WS + POS pipeline on sample text")
    p_demo.add_argument(
        "--text",
        default="Tôi là sinh viên trường đại học Công nghệ .",
        help="Unsegmented Vietnamese sentence to segment and POS-tag",
    )
    p_demo.add_argument(
        "--ws-mode",
        choices=[
            "hmm",
            "phobert",
            "hybrid",
            "cascade",
            "cascade_qwen",
            "cascade_llama",
            "qwen",
            "llama",
            "pyvi",
            "underthesea",
            "llm",
        ],
        default="phobert",
    )
    p_demo.add_argument(
        "--pos-mode",
        choices=["hmm", "phobert", "pyvi", "underthesea", "qwen", "llama", "llm"],
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

    elif args.command == "benchmark-llm":
        if args.strategy:
            config.setdefault("llm", {}).setdefault("prompting", {})["strategy"] = args.strategy
        results = run_full_evaluation(config, include_phobert=True)
        llm_rows = [r for r in results if any(k in r["model"] for k in ("qwen", "llama", "phobert", "hmm"))]
        print("=== Qwen vs. Llama vs. PhoBERT Benchmark ===")
        print_evaluation_table(llm_rows)

        ws_splits_path = paths_cfg.get("ws_splits", "data/processed/ws_splits.json")
        if os.path.exists(ws_splits_path):
            with open(ws_splits_path, "r", encoding="utf-8") as f:
                ws_splits = json.load(f)
            all_seg_sents = [
                s["segmented_text"]
                for split_list in ws_splits.values()
                for s in split_list
            ]
            models_cfg = config.get("llm", {}).get("models", {})
            fertility = compare_tokenization_fertility(
                sentences=all_seg_sents,
                qwen_tokenizer_id=models_cfg.get("qwen", {}).get("tokenizer_id", "Qwen/Qwen2.5-1.5B-Instruct"),
                llama_tokenizer_id=models_cfg.get("llama", {}).get("tokenizer_id", "Xenova/llama3-tokenizer"),
                phobert_tokenizer_id=config.get("phobert", {}).get("model_name", "vinai/phobert-base"),
            )
            print("\n=== Vietnamese Tokenization Tax (Subword Fertility) ===")
            for fam, info in fertility.items():
                if "tokens_per_syllable" in info:
                    print(
                        f"  {fam:<10} ({info['tokenizer_id']}): "
                        f"{info['tokens_per_syllable']:.4f} subwords/syllable | "
                        f"{info['tokens_per_word']:.4f} subwords/word "
                        f"(total={info['total_subwords']} subwords)"
                    )
                else:
                    print(f"  {fam:<10} ({info['tokenizer_id']}): unavailable ({info.get('error', '')[:60]})")

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
