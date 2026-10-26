"""
HMM part-of-speech tagger for Vietnamese.

Input is word-segmented text (multi-syllable words joined by '_'), e.g.
    "Học_sinh đi học ."  ->  [("Học_sinh", "N"), ("đi", "V"), ("học", "V"), (".", "CH")]

Tagset follows VLSP / VietTreebank:
    N  noun            Np proper noun      Nc classifier      Nu unit noun
    V  verb            A  adjective        P  pronoun         R  adverb
    L  determiner      M  numeral          E  preposition     C  conjunction
    CC coord. conj.    I  interjection     T  particle        Y  abbreviation
    Z  bound morpheme  X  unknown          CH punctuation

Training data formats (auto-detected):
    1. One sentence per line, tokens as word/TAG:
         Học_sinh/N đi/V học/V ./CH
    2. CoNLL-style, one "word<TAB>tag" per line, blank line between sentences.
"""

import argparse
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict

START = "<s>"
END = "</s>"

PUNCT_RE = re.compile(r"^[^\w\s]+$", re.UNICODE)
NUMBER_RE = re.compile(r"^[+-]?\d+([.,]\d+)*%?$")


def normalize(text):
    """Vietnamese diacritics can be precomposed or combining; unify to NFC."""
    return unicodedata.normalize("NFC", text)


def word_shape(word):
    """Coarse class used to score out-of-vocabulary words."""
    if PUNCT_RE.match(word):
        return "PUNCT"
    if NUMBER_RE.match(word):
        return "NUM"
    if any(ch.isdigit() for ch in word):
        return "HAS_DIGIT"
    syllables = word.split("_")
    if word.isupper() and len(word) > 1:
        return "ALL_CAPS"
    if all(s[:1].isupper() for s in syllables if s):
        return "CAP_MULTI" if len(syllables) > 1 else "CAP"
    if word[:1].isupper():
        return "CAP_PARTIAL"
    return "LOWER_MULTI" if len(syllables) > 1 else "LOWER"


def read_corpus(path):
    """Returns a list of sentences, each a list of (word, tag) pairs."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [normalize(line.rstrip("\n")) for line in f]

    is_conll = any("\t" in line for line in lines[:50])
    sentences = []

    if is_conll:
        current = []
        for line in lines:
            if not line.strip():
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split("\t")
            current.append((parts[0], parts[-1]))
        if current:
            sentences.append(current)
    else:
        for line in lines:
            tokens = line.split()
            if not tokens:
                continue
            sentence = []
            for token in tokens:
                word, sep, tag = token.rpartition("/")
                if not sep or not word:
                    raise ValueError(f"Token without /TAG in {path}: {token!r}")
                sentence.append((word, tag))
            sentences.append(sentence)

    return sentences


class HMMPosTagger:
    def __init__(self, smoothing=0.1):
        self.smoothing = smoothing
        self.tags = []
        self.transitions = {}      # log P(tag | prev_tag)
        self.emissions = {}        # log P(word | tag)
        self.unk_emissions = {}    # log P(shape | tag), learned from rare words
        self.tag_dictionary = {}   # word -> tags seen in training

    @staticmethod
    def _key(word):
        return word.lower()

    def train(self, sentences):
        trans_counts = defaultdict(Counter)
        emit_counts = defaultdict(Counter)
        word_counts = Counter()

        for sentence in sentences:
            prev = START
            for word, tag in sentence:
                key = self._key(word)
                trans_counts[prev][tag] += 1
                emit_counts[tag][key] += 1
                word_counts[key] += 1
                prev = tag
            trans_counts[prev][END] += 1

        self.tags = sorted(emit_counts)
        k = self.smoothing

        next_states = self.tags + [END]
        for prev in [START] + self.tags:
            counts = trans_counts[prev]
            total = sum(counts.values()) + k * len(next_states)
            self.transitions[prev] = {
                t: math.log((counts[t] + k) / total) for t in next_states
            }

        vocab_size = len(word_counts) + 1  # +1 reserves mass for unseen words
        for tag in self.tags:
            counts = emit_counts[tag]
            total = sum(counts.values()) + k * vocab_size
            self.emissions[tag] = {w: math.log((c + k) / total) for w, c in counts.items()}
            self.emissions[tag][None] = math.log(k / total)

        # Words seen once behave most like unseen words, so their shape
        # distribution per tag approximates P(shape | tag) for OOV tokens.
        shape_counts = defaultdict(Counter)
        for sentence in sentences:
            for word, tag in sentence:
                if word_counts[self._key(word)] == 1:
                    shape_counts[tag][word_shape(word)] += 1

        shapes = sorted({s for c in shape_counts.values() for s in c} | {"LOWER"})
        for tag in self.tags:
            counts = shape_counts[tag]
            total = sum(counts.values()) + len(shapes)
            self.unk_emissions[tag] = {s: math.log((counts[s] + 1) / total) for s in shapes}
            self.unk_emissions[tag][None] = math.log(1 / total)

        tag_dict = defaultdict(set)
        for sentence in sentences:
            for word, tag in sentence:
                tag_dict[self._key(word)].add(tag)
        self.tag_dictionary = {w: sorted(t) for w, t in tag_dict.items()}

        return self

    def _emission(self, tag, word):
        key = self._key(word)
        if key in self.tag_dictionary:
            return self.emissions[tag].get(key, self.emissions[tag][None])
        table = self.unk_emissions[tag]
        return table.get(word_shape(word), table[None])

    def _candidate_tags(self, word):
        """Restrict known words to tags they were seen with; speeds up decoding."""
        return self.tag_dictionary.get(self._key(word), self.tags)

    def tag(self, words):
        """Viterbi decoding in log space. `words` is a list of segmented words."""
        if not words:
            return []
        words = [normalize(w) for w in words]

        # viterbi[i][tag] = (best log prob ending in tag at position i, backpointer)
        viterbi = [{}]
        for tag in self._candidate_tags(words[0]):
            score = self.transitions[START][tag] + self._emission(tag, words[0])
            viterbi[0][tag] = (score, None)

        for i in range(1, len(words)):
            viterbi.append({})
            for tag in self._candidate_tags(words[i]):
                emit = self._emission(tag, words[i])
                best_prev, best_score = max(
                    ((prev, score + self.transitions[prev][tag])
                     for prev, (score, _) in viterbi[i - 1].items()),
                    key=lambda item: item[1],
                )
                viterbi[i][tag] = (best_score + emit, best_prev)

        last_tag = max(
            viterbi[-1],
            key=lambda t: viterbi[-1][t][0] + self.transitions[t][END],
        )

        path = [last_tag]
        for i in range(len(words) - 1, 0, -1):
            path.append(viterbi[i][path[-1]][1])
        path.reverse()

        return list(zip(words, path))

    def tag_text(self, text):
        return self.tag(normalize(text).split())

    def evaluate(self, sentences):
        correct = total = oov_correct = oov_total = 0
        per_tag = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

        for sentence in sentences:
            words = [w for w, _ in sentence]
            gold = [t for _, t in sentence]
            predicted = [t for _, t in self.tag(words)]

            for word, g, p in zip(words, gold, predicted):
                total += 1
                is_oov = self._key(word) not in self.tag_dictionary
                if is_oov:
                    oov_total += 1
                if g == p:
                    correct += 1
                    per_tag[g]["tp"] += 1
                    if is_oov:
                        oov_correct += 1
                else:
                    per_tag[p]["fp"] += 1
                    per_tag[g]["fn"] += 1

        report = {}
        for tag, c in sorted(per_tag.items()):
            precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
            recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            report[tag] = {"precision": precision, "recall": recall, "f1": f1,
                           "support": c["tp"] + c["fn"]}

        return {
            "accuracy": correct / total if total else 0.0,
            "oov_accuracy": oov_correct / oov_total if oov_total else None,
            "oov_rate": oov_total / total if total else 0.0,
            "tokens": total,
            "per_tag": report,
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        def encode(table):
            return {t: {("<UNK>" if k is None else k): v for k, v in d.items()}
                    for t, d in table.items()}

        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "smoothing": self.smoothing,
                "tags": self.tags,
                "transitions": self.transitions,
                "emissions": encode(self.emissions),
                "unk_emissions": encode(self.unk_emissions),
                "tag_dictionary": self.tag_dictionary,
            }, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        def decode(table):
            return {t: {(None if k == "<UNK>" else k): v for k, v in d.items()}
                    for t, d in table.items()}

        tagger = cls(smoothing=data["smoothing"])
        tagger.tags = data["tags"]
        tagger.transitions = data["transitions"]
        tagger.emissions = decode(data["emissions"])
        tagger.unk_emissions = decode(data["unk_emissions"])
        tagger.tag_dictionary = data["tag_dictionary"]
        return tagger


def split_data(sentences, ratio):
    cut = int(len(sentences) * ratio)
    return sentences[:cut], sentences[cut:]


DEMO_CORPUS = """\
Học_sinh/N đi/V học/V ./CH
Tôi/P là/V sinh_viên/N trường/N Đại_học/N Bách_khoa/Np ./CH
Hà_Nội/Np là/V thủ_đô/N của/E Việt_Nam/Np ./CH
Chúng_tôi/P đang/R học/V tiếng/N Việt/Np ở/E Huế/Np ./CH
Cô_ấy/P mua/V ba/M quyển/Nc sách/N mới/A ./CH
Anh/P ấy/T rất/R thích/V ăn/V phở/N bò/N ./CH
Hôm_nay/N trời/N đẹp/A và/CC mát/A ./CH
Mẹ/N tôi/P nấu/V cơm/N cho/E cả/L nhà/N ăn/V ./CH
Những/L con/Nc mèo/N đang/R ngủ/V trên/E ghế/N ./CH
Bộ_Y_tế/Np vừa/R công_bố/V hai/M ca/N nhiễm/V mới/A ./CH
Giá/N xăng/N tăng/V 5/M %/CH trong/E tháng/N này/P ./CH
Ông/N Nguyễn_Văn_An/Np đã/R ký/V hợp_đồng/N với/E công_ty/N ./CH
Các/L em/N học_sinh/N chăm_chỉ/A làm/V bài_tập/N ./CH
Bệnh_viện/N Chợ_Rẫy/Np tiếp_nhận/V nhiều/A bệnh_nhân/N ./CH
Tôi/P không/R biết/V anh/P ấy/T ở/V đâu/P ?/CH
Đội_tuyển/N Việt_Nam/Np đã/R thắng/V Thái_Lan/Np 2/M -/CH 1/M ./CH
Chính_phủ/N ban_hành/V nghị_định/N mới/A về/E thuế/N ./CH
Nhà/N của/E tôi/P rất/R nhỏ/A nhưng/C ấm_cúng/A ./CH
Bạn/P có/V muốn/V đi/V xem/V phim/N không/T ?/CH
Trời/N ơi/I ,/CH đẹp/A quá/R !/CH
"""


def demo():
    sentences = [
        [tuple(tok.rsplit("/", 1)) for tok in line.split()]
        for line in normalize(DEMO_CORPUS).splitlines() if line.strip()
    ]
    tagger = HMMPosTagger(smoothing=0.1).train(sentences)

    for text in [
        "Sinh_viên đang học tiếng Việt .",
        "Bà Trần_Thị_Mai mua hai con gà ở chợ Bến_Thành .",
        "Công_ty vừa công_bố 3 sản_phẩm mới .",
    ]:
        print(" ".join(f"{w}/{t}" for w, t in tagger.tag_text(text)))


def main():
    parser = argparse.ArgumentParser(description="HMM POS tagger for Vietnamese")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train on a tagged corpus")
    p_train.add_argument("corpus")
    p_train.add_argument("--model", default="models/pos_hmm.json")
    p_train.add_argument("--smoothing", type=float, default=0.1)
    p_train.add_argument("--split", type=float, default=None,
                         help="train fraction; the rest is used for evaluation")

    p_eval = sub.add_parser("eval", help="evaluate on a tagged corpus")
    p_eval.add_argument("corpus")
    p_eval.add_argument("--model", default="models/pos_hmm.json")

    p_tag = sub.add_parser("tag", help="tag segmented text")
    p_tag.add_argument("text", nargs="?", help="omit to read lines from stdin")
    p_tag.add_argument("--model", default="models/pos_hmm.json")

    sub.add_parser("demo", help="train on a tiny built-in corpus and tag examples")

    args = parser.parse_args()

    if args.command == "demo":
        demo()

    elif args.command == "train":
        sentences = read_corpus(args.corpus)
        train_set, test_set = (split_data(sentences, args.split)
                               if args.split else (sentences, []))
        tagger = HMMPosTagger(smoothing=args.smoothing).train(train_set)
        tagger.save(args.model)
        print(f"Trained on {len(train_set)} sentences, {len(tagger.tags)} tags -> {args.model}")
        if test_set:
            print_report(tagger.evaluate(test_set))

    elif args.command == "eval":
        tagger = HMMPosTagger.load(args.model)
        print_report(tagger.evaluate(read_corpus(args.corpus)))

    elif args.command == "tag":
        import sys
        tagger = HMMPosTagger.load(args.model)
        lines = [args.text] if args.text else sys.stdin
        for line in lines:
            if line.strip():
                print(" ".join(f"{w}/{t}" for w, t in tagger.tag_text(line)))


def print_report(result):
    print(f"Accuracy:     {result['accuracy']:.4f}  ({result['tokens']} tokens)")
    if result["oov_accuracy"] is not None:
        print(f"OOV accuracy: {result['oov_accuracy']:.4f}  (OOV rate {result['oov_rate']:.2%})")
    print(f"{'tag':<5}{'prec':>8}{'rec':>8}{'f1':>8}{'n':>8}")
    for tag, m in result["per_tag"].items():
        print(f"{tag:<5}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}{m['support']:>8}")


if __name__ == "__main__":
    main()
