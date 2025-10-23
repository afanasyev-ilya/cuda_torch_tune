from datasets import load_dataset
import os

# need to do export HF_ENDPOINT="https://hf-mirror.com" before running this

ROWS_LIMIT = 1000

# ~5.1M Python files; stream so you don't need full disk
ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
# write a quick LM text file (one file per doc, or join with separators)
with open("input.txt", "w", encoding="utf-8") as f:
    for i, row in enumerate(ds):
        f.write(row["content"] + "\n\n")
        if i == ROWS_LIMIT: break   # sample to start
