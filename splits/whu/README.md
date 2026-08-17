# WHU-OPT-SAR official split

`official_train.txt` and `official_test.txt` contain the published 80/20
WHU-OPT-SAR division.  The faithful reproduction deliberately uses all 80
training images and evaluates the 20-image test split every five epochs because
that is what the released MM-DINO training code does.

Selecting checkpoints on the test split is methodologically unsuitable for new
research.  It is retained here only to reproduce the public implementation; a
future clean train/validation/test protocol must live in a separate experiment.

`development_train.txt` and `development_val.txt` define that separate v1
protocol. They split the 80 published Train scenes into 64/16 scenes while
keeping every map-sheet group (the filename prefix before the final three-digit
tile id) entirely on one side. `development_manifest.json` records the GT-only
class-balance objective and canonical LF-content hashes. The official 20-scene
Test manifest was not read while selecting this split and remains reserved for
one final evaluation after a method is frozen.
