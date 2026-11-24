#!/bin/sh
# Full ablation + baseline suite. Usage: sh scripts/run_ablation.sh "--dataset ninapro_db2 --data_root /data/DB2" db2
A="$1"; D="$2"
python scripts/run_loso.py $A --model ridge                         --tag ${D}_ridge
python scripts/run_loso.py $A --model lstm                          --tag ${D}_lstm
python scripts/run_loso.py $A                                       --tag ${D}_full
python scripts/run_loso.py $A --no-use_dyn                          --tag ${D}_no_dyn
python scripts/run_loso.py $A --no-use_syn                          --tag ${D}_no_syn
python scripts/run_loso.py $A --no-use_shift --no-shift_aug         --tag ${D}_no_shift
python scripts/run_loso.py $A --no-use_film                         --tag ${D}_no_film
python scripts/run_loso.py $A --no-use_dann                         --tag ${D}_no_dann
python scripts/run_loso.py $A --no-use_nll                          --tag ${D}_no_nll
python scripts/compare.py --ref ${D}_full --tags ${D}_ridge ${D}_lstm ${D}_no_dyn ${D}_no_syn ${D}_no_shift ${D}_no_film ${D}_no_dann ${D}_no_nll
