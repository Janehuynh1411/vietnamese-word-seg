import os
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
        client = LLMWordSegmenterClient()
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


if __name__ == "__main__":
    unittest.main()
