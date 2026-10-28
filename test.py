import json
import math
import os
import tempfile
import unittest

from data.dataset_converter import (
    normalize_pos_tag,
    parse_label_studio_task,
    parse_pos_line,
    segmented_words_to_ws_tags,
    ws_tags_to_segmented_words,
)
from eval.metrics import (
    evaluate_pos_tagging,
    evaluate_word_segmentation,
    segmented_sentence_to_spans,
)
from llm.client import LLMWordSegmenterClient
from llm.phobert import align_labels_with_tokens
from llm.prompt import (
    build_pos_messages,
    build_ws_messages,
    parse_llm_pos_output,
    parse_llm_ws_output,
)


class FakePhoBERTTokenizer:
    """Lightweight mock of PhoBERT's BPE tokenizer for fast unit testing."""

    bos_token_id = 0
    pad_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    unk_token = "<unk>"

    def tokenize(self, token: str):
        # Simulate splitting long compound words into BPE subwords with '@@'
        if token == "sinh_viên":
            return ["sinh_@@", "viên"]
        return [token]

    def convert_tokens_to_ids(self, subwords):
        return [10 + idx for idx, _ in enumerate(subwords)]


class TestVietnameseWordSegPipeline(unittest.TestCase):
    def test_ws_tags_roundtrip(self):
        words = ["Tôi", "là", "sinh_viên", "trường", "đại_học", "Công_nghệ", "."]
        syllables, tags = segmented_words_to_ws_tags(words)
        self.assertEqual(syllables, ["Tôi", "là", "sinh", "viên", "trường", "đại", "học", "Công", "nghệ", "."])
        self.assertEqual(tags, ["B", "B", "B", "I", "B", "B", "I", "B", "I", "B"])
        reconstructed = ws_tags_to_segmented_words(syllables, tags)
        self.assertEqual(reconstructed, words)

    def test_pos_tag_normalization(self):
        self.assertEqual(normalize_pos_tag("F (Dấu câu / Punctuation)"), "CH")
        self.assertEqual(normalize_pos_tag("F"), "CH")
        self.assertEqual(normalize_pos_tag("N (Danh từ / Noun)"), "N")
        self.assertEqual(normalize_pos_tag("Np (Tên riêng / Proper Noun)"), "Np")

    def test_parse_pos_line(self):
        parsed = parse_pos_line("Ẩm_thực/N làm/V thăng_hoa/V ./F")
        self.assertIsNotNone(parsed)
        words, tags = parsed
        self.assertEqual(words, ["Ẩm_thực", "làm", "thăng_hoa", "."])
        self.assertEqual(tags, ["N", "V", "V", "CH"])

    def test_label_studio_task_dedup_and_parse(self):
        task = {
            "data": {"text": "Ẩm thực làm đẹp ."},
            "annotations": [
                {
                    "result": [
                        {"value": {"start": 0, "end": 7, "text": "Ẩm thực", "labels": ["N (Danh từ / Noun)"]}},
                        # Overlapping duplicate span should be deduplicated
                        {"value": {"start": 0, "end": 7, "text": "Ẩm thực", "labels": ["N (Danh từ / Noun)"]}},
                        {"value": {"start": 8, "end": 11, "text": "làm", "labels": ["V (Động từ / Verb)"]}},
                        {"value": {"start": 12, "end": 15, "text": "đẹp", "labels": ["A (Tính từ / Adjective)"]}},
                        {"value": {"start": 16, "end": 17, "text": ".", "labels": ["F (Dấu câu / Punctuation)"]}},
                    ]
                }
            ],
        }
        words, pos_tags = parse_label_studio_task(task, use_silver_fallback=False)
        self.assertEqual(words, ["Ẩm_thực", "làm", "đẹp", "."])
        self.assertEqual(pos_tags, ["N", "V", "A", "CH"])

    def test_bpe_subword_label_alignment(self):
        tokenizer = FakePhoBERTTokenizer()
        tokens = ["Tôi", "sinh_viên", "."]
        labels = ["P", "N", "CH"]
        label2id = {"P": 0, "N": 1, "CH": 2}

        aligned = align_labels_with_tokens(
            tokenizer=tokenizer,
            tokens=tokens,
            labels=labels,
            label2id=label2id,
            max_length=32,
        )
        # <s> (None, -100), Tôi (0, 0), sinh_@@ (1, 1), viên (None, -100), . (2, 2), </s> (None, -100)
        self.assertEqual(aligned["word_ids"], [None, 0, 1, None, 2, None])
        self.assertEqual(aligned["labels"], [-100, 0, 1, -100, 2, -100])

    def test_word_segmentation_and_oov_metrics(self):
        gold = ["Tôi là sinh_viên đại_học ."]
        pred = ["Tôi là sinh_viên đại học ."]
        spans = segmented_sentence_to_spans(gold[0])
        self.assertEqual(spans, [(0, 1, "Tôi"), (1, 2, "là"), (2, 4, "sinh_viên"), (4, 6, "đại_học"), (6, 7, ".")])

        metrics = evaluate_word_segmentation(gold, pred, train_vocab={"tôi", "là", "."})
        # Gold has 5 words (4 matched: Tôi, là, sinh_viên, .), Pred has 6 words
        self.assertAlmostEqual(metrics["precision"], 4 / 6, places=4)
        self.assertAlmostEqual(metrics["recall"], 4 / 5, places=4)
        # OOV words in gold: sinh_viên (matched) and đại_học (missed) -> 1/2 = 0.5
        self.assertAlmostEqual(metrics["oov_recall"], 0.5, places=4)

    def test_pos_evaluation_metrics(self):
        gold = [[("Tôi", "P"), ("là", "V"), ("sinh_viên", "N")]]
        pred = [[("Tôi", "P"), ("là", "V"), ("sinh_viên", "V")]]
        res = evaluate_pos_tagging(gold, pred, train_vocab={"tôi", "là"})
        self.assertAlmostEqual(res["accuracy"], 2 / 3, places=4)
        self.assertAlmostEqual(res["oov_accuracy"], 0.0, places=4)

    def test_llm_prompt_and_parser(self):
        msgs = build_ws_messages("Tôi là sinh viên .", strategy="chain-of-thought", ambiguous_indices=[2, 3])
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("sinh, viên", msgs[-1]["content"])

        parsed_ws = parse_llm_ws_output(
            "Step 1: sinh viên is a compound noun.\nOUTPUT: Tôi là sinh_viên .",
            "Tôi là sinh viên .",
        )
        self.assertEqual(parsed_ws, "Tôi là sinh_viên .")

        pos_msgs = build_pos_messages("Tôi là sinh_viên .", strategy="few-shot")
        self.assertTrue(len(pos_msgs) >= 3)
        parsed_pos = parse_llm_pos_output("OUTPUT: Tôi/P là/V sinh_viên/N ./CH", ["Tôi", "là", "sinh_viên", "."])
        self.assertEqual(parsed_pos, [("Tôi", "P"), ("là", "V"), ("sinh_viên", "N"), (".", "CH")])

    def test_llm_client_offline_fallback(self):
        for family in ("qwen", "llama"):
            client = LLMWordSegmenterClient(model_family=family, backend="offline")
            seg = client.segment("Tôi là sinh viên .")
            self.assertEqual(len(seg.replace("_", " ").split()), 5)
            tagged = client.tag_segmented(seg)
            self.assertEqual(len(tagged), len(seg.split()))

    def test_hmm_viterbi_confidence(self):
        if os.path.exists("models/hmm_weights.json"):
            from hmm.viterbi import ViterbiSegmenter

            seg_model = ViterbiSegmenter("models/hmm_weights.json")
            seg_text, tags, confs = seg_model.segment_with_confidence("Ẩm thực làm thăng hoa")
            self.assertEqual(len(tags), 5)
            self.assertEqual(len(confs), 5)
            self.assertTrue(all(0.0 <= c <= 1.0 for c in confs))

    def test_hmm_logspace_posteriors_and_unk_smoothing(self):
        from hmm.train import train_hmm
        from hmm.viterbi import ViterbiSegmenter

        corpus = "\n".join(["học_sinh đi học"] * 20 + ["đại_học công_nghệ"] * 10)
        with tempfile.TemporaryDirectory() as tmp:
            train_path = os.path.join(tmp, "train.txt")
            weights_path = os.path.join(tmp, "hmm.json")
            with open(train_path, "w", encoding="utf-8") as handle:
                handle.write(corpus)
            model = train_hmm(train_path)
            for state in ("B", "I"):
                self.assertIn("<UNK>", model["emissions"][state])
                self.assertAlmostEqual(sum(model["emissions"][state].values()), 1.0, places=6)
            with open(weights_path, "w", encoding="utf-8") as handle:
                json.dump(model, handle)

            segmenter = ViterbiSegmenter(weights_path)
            sentence = " ".join(["học"] * 40)
            _, tags, confs = segmenter.segment_with_confidence(sentence)
            self.assertEqual(len(tags), 40)
            self.assertEqual(len(confs), 40)
            self.assertTrue(math.isfinite(segmenter.last_path_log_score))
            for posterior in segmenter.last_posteriors:
                self.assertAlmostEqual(posterior["B"] + posterior["I"], 1.0, places=5)
            self.assertTrue(all(0.0 <= conf <= 1.0 for conf in confs))

            # Emissions this small underflow a linear-space product over 40 syllables.
            tiny = {
                "states": ["B", "I"],
                "transitions": {
                    "B": {"B": 0.5, "I": 0.5},
                    "I": {"B": 0.5, "I": 0.5},
                },
                "emissions": {
                    "B": {"học": 1e-12, "<UNK>": 1e-12},
                    "I": {"học": 1e-12, "<UNK>": 1e-12},
                },
                "unk_token": "<UNK>",
            }
            tiny_path = os.path.join(tmp, "tiny.json")
            with open(tiny_path, "w", encoding="utf-8") as handle:
                json.dump(tiny, handle)
            tiny_segmenter = ViterbiSegmenter(tiny_path)
            _, tiny_tags, _ = tiny_segmenter.segment_with_confidence(sentence)
            self.assertEqual(len(tiny_tags), 40)
            self.assertTrue(math.isfinite(tiny_segmenter.last_path_log_score))
            self.assertLess(tiny_segmenter.last_path_log_score, 0.0)

    def test_llama_tokenizer_preset_matches_model(self):
        import inspect

        from llm.client import MODEL_PRESETS, compare_tokenization_fertility

        llama = MODEL_PRESETS["llama"]
        qwen = MODEL_PRESETS["qwen"]
        self.assertEqual(llama["tokenizer_id"], llama["hf_model"])
        self.assertEqual(llama["tokenizer_id"], "meta-llama/Llama-3.2-1B-Instruct")
        self.assertNotEqual(llama["tokenizer_id"], qwen["tokenizer_id"])
        default = inspect.signature(compare_tokenization_fertility).parameters["llama_tokenizer_id"].default
        self.assertEqual(default, "meta-llama/Llama-3.2-1B-Instruct")

    def test_dpo_constraints_in_ws_prompt_and_parser(self):
        msgs = build_ws_messages(
            "Tôi là sinh viên .",
            strategy="zero-shot",
            preferred_compounds=["sinh_viên"],
            rejected_compounds=["đại_học"],
        )
        content = msgs[-1]["content"]
        self.assertIn("sinh_viên", content)
        self.assertIn("đại_học", content)
        self.assertIn("Prefer these compound joins", content)
        self.assertIn("Do not join these rejected boundaries", content)

        parsed = parse_llm_ws_output("OUTPUT: Tôi là sinh_viên.", "Tôi là sinh viên .")
        self.assertIn("sinh_viên", parsed)
        self.assertNotEqual(parsed, "Tôi là sinh viên .")

        combining = "Tôi là sinh vie\u0302n ."
        parsed_nfc = parse_llm_ws_output("OUTPUT: Tôi là sinh_viên .", combining)
        self.assertEqual(parsed_nfc, "Tôi là sinh_viên .")

        dropped = parse_llm_ws_output("OUTPUT: Tôi là sinh_viên", "Tôi là sinh viên .")
        self.assertEqual(dropped, "Tôi là sinh viên .")

    def test_rejected_compound_filter(self):
        client = LLMWordSegmenterClient(model_family="llama", backend="offline", dpo_memory_path="missing-memory.json")
        client.rejected_compounds = {"sinh_viên"}
        syllables = ["Tôi", "là", "sinh", "viên"]
        tags = ["B", "B", "B", "I"]
        filtered = client._apply_dpo_rejected_filter(syllables, tags)
        self.assertEqual(filtered, ["B", "B", "B", "B"])

    def test_upos_mapping_and_punctuation_spans(self):
        self.assertEqual(normalize_pos_tag("NOUN"), "N")
        self.assertEqual(normalize_pos_tag("VERB"), "V")
        self.assertEqual(normalize_pos_tag("ADJ"), "A")
        self.assertEqual(normalize_pos_tag("PROPN"), "Np")
        self.assertEqual(normalize_pos_tag("PUNCT"), "CH")
        self.assertEqual(normalize_pos_tag("AUX"), "V")

        gold = [[("Tôi", "PRON"), ("là", "VERB"), ("sinh_viên", "NOUN")]]
        pred = [[("Tôi", "P"), ("là", "V"), ("sinh_viên", "N")]]
        scores = evaluate_pos_tagging(gold, pred)
        self.assertAlmostEqual(scores["accuracy"], 1.0, places=4)

        spans = segmented_sentence_to_spans("Tôi là sinh_viên.")
        self.assertEqual(
            spans,
            [(0, 1, "Tôi"), (1, 2, "là"), (2, 4, "sinh_viên"), (4, 5, ".")],
        )
        detached = evaluate_word_segmentation(
            ["Tôi là sinh_viên ."],
            ["Tôi là sinh_viên."],
        )
        self.assertAlmostEqual(detached["f1_score"], 1.0, places=4)

    def test_dpo_loss_prefers_chosen_and_builds_memory(self):
        import torch

        from data.dpo_builder import build_dpo_dataset
        from llm.dpo_trainer import (
            assemble_dpo_objective,
            build_llm_preference_memory,
            load_preference_pairs,
            sequence_log_probs_from_logits,
        )

        chosen = torch.zeros(2, 4, dtype=torch.long)
        rejected = torch.ones(2, 4, dtype=torch.long)
        chosen[0, 0] = -100
        favored = torch.tensor([[[4.0, 0.0]] * 4, [[4.0, 0.0]] * 4])
        disfavored = torch.tensor([[[0.0, 4.0]] * 4, [[0.0, 4.0]] * 4])
        reference = torch.zeros_like(favored)
        sft = torch.tensor(0.0)
        good_loss, _, good_margin = assemble_dpo_objective(favored, reference, chosen, rejected, sft)
        bad_loss, _, bad_margin = assemble_dpo_objective(disfavored, reference, chosen, rejected, sft)
        self.assertLess(float(good_loss), float(bad_loss))
        self.assertGreater(float(good_margin), 0.0)
        self.assertLess(float(bad_margin), 0.0)
        logps = sequence_log_probs_from_logits(favored, chosen)
        self.assertEqual(tuple(logps.shape), (2,))
        self.assertTrue(torch.isfinite(logps).all())

        with tempfile.TemporaryDirectory() as tmp:
            splits_path = os.path.join(tmp, "ws_splits.json")
            splits = {
                "train": [
                    {
                        "syllables": ["Ẩm", "thực", "làm"],
                        "tags": ["B", "I", "B"],
                        "raw_text": "Ẩm thực làm",
                        "segmented_text": "Ẩm_thực làm",
                    }
                ],
                "val": [],
            }
            with open(splits_path, "w", encoding="utf-8") as handle:
                json.dump(splits, handle)
            summary = build_dpo_dataset(
                ws_splits_path=splits_path,
                hmm_weights_path=os.path.join(tmp, "missing.json"),
                phobert_ckpt_path=None,
                output_dir=tmp,
            )
            self.assertGreaterEqual(summary["pairs"], 1)
            self.assertTrue(os.path.exists(summary["label_studio_config"]))
            pairs = load_preference_pairs(summary["jsonl"])
            memory_path = os.path.join(tmp, "llm_dpo_memory.json")
            memory = build_llm_preference_memory(pairs, memory_path)
            self.assertIn("ẩm_thực", memory["preferred_compounds"])
            self.assertGreaterEqual(memory["preferred_count"], 1)

    def test_hmm_tier_records_latency(self):
        import yaml

        from hmm.train import train_hmm
        from pipeline import VietnameseNLPPipeline

        with tempfile.TemporaryDirectory() as tmp:
            train_path = os.path.join(tmp, "train.txt")
            weights_path = os.path.join(tmp, "hmm.json")
            config_path = os.path.join(tmp, "config.yaml")
            with open(train_path, "w", encoding="utf-8") as handle:
                handle.write("học_sinh đi học\n")
            model = train_hmm(train_path)
            with open(weights_path, "w", encoding="utf-8") as handle:
                json.dump(model, handle)
            config = {
                "hmm": {"ws_weights": weights_path, "pos_weights": os.path.join(tmp, "missing-pos.json")},
                "routing": {"hmm_confidence_threshold": 0.85, "phobert_confidence_threshold": 0.70},
                "phobert": {},
                "llm": {},
                "paths": {},
            }
            with open(config_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle)
            pipe = VietnameseNLPPipeline(ws_mode="hmm", pos_mode="hmm", config_path=config_path)
            result = pipe.process("học sinh đi học")
            self.assertIn("_", result["segmented_text"])
            self.assertGreater(result["routing_stats"]["tier1_ms"], 0.0)
            self.assertEqual(result["routing_stats"]["tier1_hmm_syllables"], 4)
            self.assertIn("tier2_ms", result["routing_stats"])
            self.assertIn("tier3_ms", result["routing_stats"])


if __name__ == "__main__":
    unittest.main()
