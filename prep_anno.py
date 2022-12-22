import json
import datasets
from tqdm.auto import tqdm

DSET_FILES = {
    "train": "/home/mowentao/scratch/ChID_baseline/dataset/train_data.json",
    "val": "/home/mowentao/scratch/ChID_baseline/dataset/dev_data.json",
    # "test": "/home/mowentao/scratch/ChID_baseline/dataset/test_data.json",
}

# lines = []
# for dset_file in DSET_FILES:
#     lines += open(dset_file, "r").readlines()

# lines = map(json.loads)
dset = datasets.load_dataset("json", data_files=DSET_FILES)
dset = datasets.concatenate_datasets(list(dset.values()))

idiom_vocab = set()
for cand in tqdm(dset["candidates"]):
    for c in cand:
        idiom_vocab.update(c)

print(len(idiom_vocab))

json.dump(sorted(idiom_vocab), open("idiom_vocab.json", "w"))
