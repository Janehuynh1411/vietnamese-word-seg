import os
import json
from underthesea import sent_tokenize

# Configuration
input_folder = "./data/raw"
output_dir = "./data/processed"
MAX_SENTENCES_PER_FILE = 2000  # Adjust this number to make files smaller or larger

tasks = []
file_counter = 1
skipped_empty = 0
skipped_errors = 0


def read_text_file(file_path):
    """Read a text file, supporting UTF-16/UTF-8 encodings. Returns None for empty files."""
    with open(file_path, "rb") as f:
        raw = f.read()

    if not raw:
        return None

    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16")
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("utf-16")
            except UnicodeDecodeError:
                text = raw.decode("utf-8", errors="replace")

    if not text.strip():
        return None

    return text


def save_batch(batch_tasks, counter):
    """Helper function to save a chunk of tasks to a JSON file"""
    output_file = os.path.join(output_dir, f"label_studio_import_part{counter}.json")
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(batch_tasks, out_f, ensure_ascii=False, indent=2)
    print(f"Saved {len(batch_tasks)} sentences to: {output_file}")


# Verify the data/raw folder exists
if not os.path.exists(input_folder):
    print(f"Error: The folder '{input_folder}' does not exist.")
    exit(1)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Walk through your category subfolders
for root, dirs, files in os.walk(input_folder):
    for filename in files:
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(root, filename)
        category = os.path.basename(root)

        try:
            content = read_text_file(file_path)
        except OSError as exc:
            print(f"Warning: Could not read '{file_path}': {exc}")
            skipped_errors += 1
            continue

        if content is None:
            skipped_empty += 1
            continue

        for sentence in sent_tokenize(content):
            clean_sentence = sentence.strip()
            if not clean_sentence:
                continue

            tasks.append({
                "data": {
                    "text": clean_sentence,
                    "meta": {
                        "source_file": filename,
                        "category": category,
                    },
                }
            })

            # Once we hit the limit, save the batch and clear the list
            if len(tasks) == MAX_SENTENCES_PER_FILE:
                save_batch(tasks, file_counter)
                file_counter += 1
                tasks = []

# Save any remaining sentences that didn't perfectly fill the last batch
if tasks:
    save_batch(tasks, file_counter)

print("\nAll files successfully batched and processed!")
if skipped_empty:
    print(f"Skipped {skipped_empty} empty text file(s).")
if skipped_errors:
    print(f"Skipped {skipped_errors} unreadable file(s).")
