import os
import json
from collections import Counter, defaultdict

UNK_TOKEN = "<UNK>"

try:
    import yaml
except ImportError:
    yaml = None


def load_config(config_path="config.yaml"):
    if yaml is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {"paths": {"processed_data": "data/processed"}}

def get_tags(sentence):
    """
    Converts a segmented sentence into a list of (syllable, tag) pairs.
    Example: "Học_sinh đi học" -> [("Học", "B"), ("sinh", "I"), ("đi", "B"), ("học", "B")]
    """
    tagged_sentence = []
    words = sentence.split()
    for word in words:
        syllables = word.split('_')
        for i, syllable in enumerate(syllables):
            tag = 'B' if i == 0 else 'I'
            tagged_sentence.append((syllable.lower(), tag))
    return tagged_sentence

def train_hmm(data_path):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    states = ['B', 'I']
    vocab = set()

    # Add-one over the closed B/I transition space.
    for s1 in states:
        for s2 in states:
            transitions[s1][s2] = 1

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            tagged_data = get_tags(line.strip())
            if not tagged_data:
                continue

            for i in range(len(tagged_data)):
                syllable, tag = tagged_data[i]
                vocab.add(syllable)
                emissions[tag][syllable] += 1

                if i > 0:
                    prev_tag = tagged_data[i-1][1]
                    transitions[prev_tag][tag] += 1

    model = {
        "transitions": {},
        "emissions": {},
        "states": states,
        "unk_token": UNK_TOKEN,
    }

    for state, counts in transitions.items():
        total = sum(counts.values())
        model["transitions"][state] = {next_state: c / total for next_state, c in counts.items()}

    # Add-one over the observed syllable vocabulary plus an <UNK> bin, so unseen
    # syllables keep a real probability instead of an arbitrary 1e-10 floor.
    for state in states:
        smoothed = Counter()
        for syll in vocab:
            smoothed[syll] = emissions[state][syll] + 1
        smoothed[UNK_TOKEN] = 1
        total = sum(smoothed.values()) or 1
        model["emissions"][state] = {syll: c / total for syll, c in smoothed.items()}

    return model

if __name__ == "__main__":
    config = load_config()
    train_file = os.path.join(config['paths']['processed_data'], "train.txt")
    
    print(f"Training HMM on {train_file}")
    
    if os.path.exists(train_file):
        trained_model = train_hmm(train_file)
        
        # Save weights for viterbi.py to use
        output_path = "models/hmm_weights.json"
        os.makedirs("models", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(trained_model, f, ensure_ascii=False, indent=4)
            
        print(f"Saved HMM weights to {output_path}")
    else:
        print(f"Error: file not found.")