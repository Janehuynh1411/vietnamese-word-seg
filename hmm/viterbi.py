import json
import os

class ViterbiSegmenter:
    def __init__(self, weights_path="models/hmm_weights.json"):
        with open(weights_path, 'r', encoding='utf-8') as f:
            self.model = json.load(f)
        
        self.transitions = self.model['transitions']
        self.emissions = self.model['emissions']
        self.states = self.model['states']
        
        # Small probability for Out-of-Vocabulary (OOV) syllables
        self.min_prob = 1e-10

    def get_emission_prob(self, state, syllable):
        """Returns the probability of a syllable given a tag (B or I)."""
        return self.emissions[state].get(syllable.lower(), self.min_prob)

    def segment(self, text):
        """
        Implementation of the Viterbi Algorithm.
        Finds the most likely sequence of B/I tags for a string of syllables.
        """
        syllables = text.split()
        if not syllables:
            return ""

        # T[state][i] = max probability of being in 'state' at syllable 'i'
        T = [{state: {"prob": 0.0, "prev": None} for state in self.states} for _ in range(len(syllables))]

        # 1. Initialization (First syllable must be 'B')
        for state in self.states:
            prob = (1.0 if state == 'B' else 0.0) * self.get_emission_prob(state, syllables[0])
            T[0][state] = {"prob": prob, "prev": None}

        # 2. Recursion (Dynamic Programming)
        for i in range(1, len(syllables)):
            for current_state in self.states:
                max_prob = -1.0
                best_prev = None
                
                emission_p = self.get_emission_prob(current_state, syllables[i])
                
                for prev_state in self.states:
                    # Prob = prob of prev path * transition * emission
                    trans_p = self.transitions[prev_state].get(current_state, self.min_prob)
                    total_p = T[i-1][prev_state]["prob"] * trans_p * emission_p
                    
                    if total_p > max_prob:
                        max_prob = total_p
                        best_prev = prev_state
                
                T[i][current_state] = {"prob": max_prob, "prev": best_prev}

        # 3. Backtracking
        path = []
        max_final_prob = -1.0
        best_last_state = None

        for state in self.states:
            if T[-1][state]["prob"] > max_final_prob:
                max_final_prob = T[-1][state]["prob"]
                best_last_state = state

        current_state = best_last_state
        for i in range(len(syllables) - 1, -1, -1):
            path.insert(0, current_state)
            current_state = T[i][current_state]["prev"]

        # 4. Format Output (Add underscores based on B/I tags)
        result = []
        for i, (syllable, tag) in enumerate(zip(syllables, path)):
            if tag == 'B' and i > 0:
                result.append(" " + syllable)
            elif tag == 'I':
                result.append("_" + syllable)
            else:
                result.append(syllable)
        
        return "".join(result)

    def segment_with_confidence(self, text):
        """
        Run Viterbi decoding and return (segmented_text, path_tags, confidences).
        Confidence at syllable i is the normalized posterior probability of the chosen
        state at position i: P(best_state) / (P('B') + P('I')), penalized to 0.0 if the
        syllable is Out-of-Vocabulary (OOV).
        """
        syllables = text.split()
        if not syllables:
            return "", [], []

        known_vocab = set(self.emissions.get('B', {}).keys()) | set(self.emissions.get('I', {}).keys())
        T = [{state: {"prob": 0.0, "prev": None} for state in self.states} for _ in range(len(syllables))]

        for state in self.states:
            prob = (1.0 if state == 'B' else 0.0) * self.get_emission_prob(state, syllables[0])
            T[0][state] = {"prob": prob, "prev": None}

        for i in range(1, len(syllables)):
            for current_state in self.states:
                max_prob = -1.0
                best_prev = None
                emission_p = self.get_emission_prob(current_state, syllables[i])
                for prev_state in self.states:
                    trans_p = self.transitions[prev_state].get(current_state, self.min_prob)
                    total_p = T[i - 1][prev_state]["prob"] * trans_p * emission_p
                    if total_p > max_prob:
                        max_prob = total_p
                        best_prev = prev_state
                T[i][current_state] = {"prob": max_prob, "prev": best_prev}

        path = []
        max_final_prob = -1.0
        best_last_state = None
        for state in self.states:
            if T[-1][state]["prob"] > max_final_prob:
                max_final_prob = T[-1][state]["prob"]
                best_last_state = state

        current_state = best_last_state
        for i in range(len(syllables) - 1, -1, -1):
            path.insert(0, current_state)
            current_state = T[i][current_state]["prev"]

        confidences = []
        for i, (syllable, chosen_state) in enumerate(zip(syllables, path)):
            if syllable.lower() not in known_vocab:
                confidences.append(0.0)
                continue
            p_b = T[i].get('B', {}).get("prob", 0.0)
            p_i = T[i].get('I', {}).get("prob", 0.0)
            denom = p_b + p_i
            if denom <= 0.0:
                confidences.append(0.5)
            else:
                chosen_p = p_b if chosen_state == 'B' else p_i
                confidences.append(chosen_p / denom)

        result = []
        for i, (syllable, tag) in enumerate(zip(syllables, path)):
            if tag == 'B' and i > 0:
                result.append(" " + syllable)
            elif tag == 'I':
                result.append("_" + syllable)
            else:
                result.append(syllable)

        return "".join(result), path, confidences

if __name__ == "__main__":
    # Test the segmenter
    if os.path.exists("models/hmm_weights.json"):
        segmenter = ViterbiSegmenter()
        test_sentence = "Học sinh học sinh học"
        segmented = segmenter.segment(test_sentence)
        print(f"Input: {test_sentence}")
        print(f"Segmented: {segmented}")
    else:
        print("Error: models/hmm_weights.json not found. Run train.py first.")