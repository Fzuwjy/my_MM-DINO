# WHU-OPT-SAR official split

`official_train.txt` and `official_test.txt` contain the published 80/20
WHU-OPT-SAR division.  The faithful reproduction deliberately uses all 80
training images and evaluates the 20-image test split every five epochs because
that is what the released MM-DINO training code does.

Selecting checkpoints on the test split is methodologically unsuitable for new
research.  It is retained here only to reproduce the public implementation; a
future clean train/validation/test protocol must live in a separate experiment.
