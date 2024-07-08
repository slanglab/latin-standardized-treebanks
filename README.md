# latin-standardized-treebanks

## Setup for converting treebanks
If you'd like to replicate the conversion process, follow these steps. We've also provided the converted files already in `data/converted_treebanks/`, so these steps are optional.

First, clone the Latin treebanks. The first is 5 harmonized UD treebanks by Gamba and Zeman (2023), and the second is the LASLA treebank (Denooz 2004).
```
cd data/original_treebanks/
git clone https://github.com/fjambe/Latin-variability.git
git clone https://github.com/CIRCSE/LASLA.git
```

Then run the conversion scripts inside `latin-standardized-treebanks/code/conversion_scripts/`:
- `convert_ud_treebanks.py`: converts treebanks to our tagset; creates two subdirectories, `ud` and `lasla` inside of `data/converted_treebanks/`.
- `make_custom_treebanks.py`: splits the converted ud treebanks inside `data/ud/` into files for each individual work, which are saved in `data/full_ud_sets/`. Also creates custom train/test splits, saved in `data/train_test_splits`.
