from datasets import load_dataset

# "roneneldan/TinyStories" has 'train' and 'validation' splits, each with a 'text' field
ds = load_dataset("roneneldan/TinyStories", split="train")  # you can also concatenate 'validation' if you want

# Write one story per line. Char-level training will just see a big stream of characters.
import os
os.makedirs("./data", exist_ok=True)
with open("./data/stories.txt", "w", encoding="utf-8") as f:
    for ex in ds:
        t = (ex["text"] or "").strip()
        if t:
            f.write(t + "\n")  # newline separates stories
print("Wrote ./data/stories.txt")
