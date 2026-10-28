import json
import math
import os

UNK_TOKEN = "<UNK>"
_NEG_INF = float("-inf")


def _log(prob: float) -> float:
    if prob <= 0.0:
        return _NEG_INF
    return math.log(prob)


def _logsumexp(values) -> float:
    finite = [value for value in values if value != _NEG_INF and not math.isnan(value)]
    if not finite:
        return _NEG_INF
    peak = max(finite)
    return peak + math.log(sum(math.exp(value - peak) for value in finite))


class ViterbiSegmenter:
    def __init__(self, weights_path="models/hmm_weights.json"):
        with open(weights_path, "r", encoding="utf-8") as f:
            self.model = json.load(f)

        self.transitions = self.model["transitions"]
        self.emissions = self.model["emissions"]
        self.states = self.model["states"]
        self.unk_token = self.model.get("unk_token", UNK_TOKEN)

        # Floor used only when a weight file has neither the syllable nor <UNK>.
        self.min_prob = 1e-10
        self.last_path_log_score = _NEG_INF
        self.last_posteriors = []

    def get_emission_prob(self, state, syllable):
        """Probability of a syllable given a B/I state, with smoothed <UNK> mass."""
        table = self.emissions.get(state, {})
        key = syllable.lower()
        if key in table:
            return table[key]
        if self.unk_token in table:
            return table[self.unk_token]
        return self.min_prob

    def _log_emission(self, state, syllable) -> float:
        return _log(self.get_emission_prob(state, syllable))

    def _log_transition(self, prev_state, current_state) -> float:
        prob = self.transitions.get(prev_state, {}).get(current_state, self.min_prob)
        return _log(prob)

    def _start_log_prob(self, state) -> float:
        # The first syllable of a word is B. I is impossible at position 0.
        return 0.0 if state == "B" else _NEG_INF

    def _viterbi(self, syllables):
        """Log-space Viterbi. Returns (path, path_log_score)."""
        states = self.states
        scores = [{state: _NEG_INF for state in states} for _ in syllables]
        backpointers = [{state: None for state in states} for _ in syllables]

        for state in states:
            scores[0][state] = self._start_log_prob(state) + self._log_emission(state, syllables[0])

        for i in range(1, len(syllables)):
            for current_state in states:
                emission = self._log_emission(current_state, syllables[i])
                best_score = _NEG_INF
                best_prev = None
                for prev_state in states:
                    total = scores[i - 1][prev_state] + self._log_transition(prev_state, current_state) + emission
                    if total > best_score:
                        best_score = total
                        best_prev = prev_state
                scores[i][current_state] = best_score
                backpointers[i][current_state] = best_prev

        best_last_state = states[0]
        best_score = scores[-1][best_last_state]
        for state in states[1:]:
            if scores[-1][state] > best_score:
                best_score = scores[-1][state]
                best_last_state = state

        path = []
        current_state = best_last_state
        for i in range(len(syllables) - 1, -1, -1):
            path.append(current_state)
            current_state = backpointers[i][current_state]
        path.reverse()
        return path, best_score

    def _forward_backward(self, syllables):
        """
        Marginal posterior γ_i(s) = P(q_i = s | O, λ) from log-space forward-backward.
        Each position is a dict {state: probability} that sums to 1.
        """
        states = list(self.states)
        n = len(syllables)
        log_alpha = [{state: _NEG_INF for state in states} for _ in range(n)]
        log_beta = [{state: _NEG_INF for state in states} for _ in range(n)]

        for state in states:
            log_alpha[0][state] = self._start_log_prob(state) + self._log_emission(state, syllables[0])

        for i in range(1, n):
            for current_state in states:
                emission = self._log_emission(current_state, syllables[i])
                terms = [
                    log_alpha[i - 1][prev_state] + self._log_transition(prev_state, current_state) + emission
                    for prev_state in states
                ]
                log_alpha[i][current_state] = _logsumexp(terms)

        for state in states:
            log_beta[n - 1][state] = 0.0

        for i in range(n - 2, -1, -1):
            for prev_state in states:
                terms = []
                for current_state in states:
                    emission = self._log_emission(current_state, syllables[i + 1])
                    terms.append(
                        self._log_transition(prev_state, current_state)
                        + emission
                        + log_beta[i + 1][current_state]
                    )
                log_beta[i][prev_state] = _logsumexp(terms)

        log_z = _logsumexp(log_alpha[n - 1][state] + log_beta[n - 1][state] for state in states)
        posteriors = []
        for i in range(n):
            logs = {state: log_alpha[i][state] + log_beta[i][state] - log_z for state in states}
            probs = {state: (math.exp(value) if value > -700 else 0.0) for state, value in logs.items()}
            total = sum(probs.values())
            if total > 0.0:
                probs = {state: value / total for state, value in probs.items()}
            posteriors.append(probs)
        return posteriors

    def _format(self, syllables, path) -> str:
        result = []
        for i, (syllable, tag) in enumerate(zip(syllables, path)):
            if tag == "B" and i > 0:
                result.append(" " + syllable)
            elif tag == "I":
                result.append("_" + syllable)
            else:
                result.append(syllable)
        return "".join(result)

    def segment(self, text):
        """
        Log-space Viterbi decoding.
        Finds the most likely sequence of B/I tags for a string of syllables.
        """
        syllables = text.split()
        if not syllables:
            self.last_path_log_score = _NEG_INF
            self.last_posteriors = []
            return ""

        path, path_log_score = self._viterbi(syllables)
        self.last_path_log_score = path_log_score
        return self._format(syllables, path)

    def segment_with_confidence(self, text):
        """
        Run Viterbi decoding and return (segmented_text, path_tags, confidences).

        The tag path is the Viterbi path. Confidence at syllable i is the
        forward-backward posterior γ_i(chosen tag), which sums with the other
        state to 1. `last_path_log_score` holds the log-probability of that path
        and `last_posteriors` holds γ_i for every state.
        """
        syllables = text.split()
        if not syllables:
            self.last_path_log_score = _NEG_INF
            self.last_posteriors = []
            return "", [], []

        path, path_log_score = self._viterbi(syllables)
        posteriors = self._forward_backward(syllables)
        confidences = [posteriors[i].get(tag, 0.0) for i, tag in enumerate(path)]

        self.last_path_log_score = path_log_score
        self.last_posteriors = posteriors
        return self._format(syllables, path), path, confidences


if __name__ == "__main__":
    if os.path.exists("models/hmm_weights.json"):
        segmenter = ViterbiSegmenter()
        test_sentence = "Học sinh học sinh học"
        segmented = segmenter.segment(test_sentence)
        print(f"Input: {test_sentence}")
        print(f"Segmented: {segmented}")
        print(f"Path log score: {segmenter.last_path_log_score}")
    else:
        print("Error: models/hmm_weights.json not found. Run train.py first.")
