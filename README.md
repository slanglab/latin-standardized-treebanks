# latin-standardized-treebanks
Code and data for: Marisa Hudspeth, Brendan O’Connor, Laure Thompson, "Latin Treebanks in Review: An Evaluation of Morphological Tagging Across Time." Machine Learning for Ancient Languages (ML4AL) Workshop, 2024.

## Directory Structure
```
├── code
│   ├── conversion_scripts
```
- Code for converting from UD/LASLA tagset to a Standard Latin tagset.
```
│   └── morph_tagging
│       └── scripts
```
- Code for finetuning LatinBERT ([Bamman and Burns, 2020]([https://doi.org/10.48550/arXiv.2009.10053](https://doi.org/10.48550/arXiv.2009.10053))) for morphological tagging.
```
├── data
│   ├── converted_treebanks
│   │   ├── lasla
│   │   ├── ud
│   │   ├── full_ud_sets
│   │   ├── train_test_splits
│   │   └── split_ids.json
```
- Directories containing UD and LASLA treebanks converted to a Standard Latin tagset:
  - `lasla`: converted files from the [LASLA repo](https://github.com/CIRCSE/LASLA/tree/main/conllup) 
  - `ud`: converted files from 5 UD treebanks. Original, UD train-test splits
  - `full_ud_sets`: individual files for each unique text in the 5 UD treebanks. No train/test splits
  - `train_test_splits`: custom train/test splits. Each unique work in LASLA and UD, has separate files for its train, dev, and test sets (whichever are applicable). 
  - `split_ids.json`: for each unique work in LASLA and UD, lists which sentence IDs belong to which train/dev/test set 
```
│   ├── original_treebanks 
```
- Directory to hold original treebanks (not included in repo, but can be downloaded)
```
└── └── metadata.csv
```
- For each unique work in LASLA and UD treebanks, lists: source treebank(s), time period, century, number of sentences. For UD texts, also includes genre labels. 

## Setup for converting treebanks
If you'd like to replicate the conversion process, follow these steps. We've also provided the converted files already in `data/converted_treebanks/`, so these steps are optional.

First, clone the Latin treebanks. The first is 5 harmonized UD treebanks by [Gamba and Zeman (2023)](https://aclanthology.org/2023.alp-1.7/), and the second is the LASLA treebank ([Denooz 2004](https://doi.org/10.1484/J.EUPHR.5.125535)). 
```
cd data/original_treebanks/
git clone https://github.com/fjambe/Latin-variability.git
git clone https://github.com/CIRCSE/LASLA.git
```

Then run the conversion scripts inside `latin-standardized-treebanks/code/conversion_scripts/`:
- `convert_ud_treebanks.py`: creates two subdirectories, `ud` and `lasla` inside of `data/converted_treebanks/`, and converts treebanks to our tagset.
- `make_custom_treebanks.py`: splits the converted ud treebanks inside `data/ud/` into files for each individual work, which are saved in `data/full_ud_sets/`. Also creates custom train/test splits, saved in `data/train_test_splits`.

## Finetuning LatinBERT for morphological tagging
Clone the [LatinBERT repo](https://github.com/dbamman/latin-bert):
```
cd code/
git clone https://github.com/dbamman/latin-bert.git
mv morph_tagging ./latin-bert/case_studies/
```

Optional: generate the tagset (we have already included the tagset file in `data/converted_treebanks/morph.tagset`):
```
cd latin-bert/case_studies/morph_tagging/scripts
python generate_tagset.py -f ../../../../../data/converted_treebanks/train_test_splits/*.conllu > morph.tagset
```

Example train command:
```
python latin_sequence_labeling.py -m train \
    --bertPath ../../../models/latin_bert \
    --tokenizerPath ../../../models/subword_tokenizer_latin/latin.subword.encoder \
    -r [LIST OF TRAINING FILEPATHS] \
    -g ../../../../../data/converted_treebanks/morph.tagset \
    -f [MODEL SAVE PATH] \
    -z fscore \
    --max_epochs [MAX_EPOCHS] \
    --load_from_checkpoint_num=-1 \
    --save_every_n=-1 
```
Options inside brackets [] are for you to fill in.

Example predict command:
```
python latin_sequence_labeling.py -m predict \
    --bertPath ../../../models/latin_bert \
    --tokenizerPath ../../../models/subword_tokenizer_latin/latin.subword.encoder \
    -g ../../../../../data/converted_treebanks/morph.tagset \
    -f [MODEL SAVE PATH]\
    -z fscore \
    -i [LIST OF TEST FILEPATHS] \
    -o [LIST OF OUTPUT FILEPATHS, PARALLEL TO TEST PATHS]
```

Our code is adapted from the [POS tagging case study code](https://github.com/dbamman/latin-bert/tree/master/case_studies/pos_tagging) for the LatinBERT paper.