# WHU-OPT-SAR official split

`official_train.txt` and `official_test.txt` contain the published 80/20
WHU-OPT-SAR division.  The faithful reproduction deliberately uses all 80
training images and evaluates the 20-image test split every five epochs because
that is what the released MM-DINO training code does.

New WHU research deliberately keeps this official 80/20 protocol for direct
comparison with published results; no additional validation split is tracked.
Because the 20-image Test split is used during development, all such results
must be labelled development-exposed rather than blind-test estimates. Method
definitions are frozen after WHU and checked without dataset-specific rescue
changes on EarthMiss and other datasets.
