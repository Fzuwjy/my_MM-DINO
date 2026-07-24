# WHU-OPT-SAR split metadata

`official_train.txt` and `official_test.txt` reproduce the 80/20 image division
published in the official
[WHU-OPT-SAR dataset repository](https://github.com/AmberHen/WHU-OPT-SAR-dataset/blob/main/The%20division%20of%20the%20dataset.txt).

Do not select checkpoints on `official_test.txt`. Generate a provisional
group-disjoint train/validation split from `official_train.txt` with
`scripts/make_grouped_split.py`, then inspect the class-pixel balance before
freezing and committing the research split.
