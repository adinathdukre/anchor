#!/bin/sh
# Download the public NHANES 2017-2018 files used by nhanes_anchor.py (about 5.5 MB).
D="$(dirname "$0")/../../data_external/nhanes_2017_2018"; mkdir -p "$D"
for f in FASTQX_J DEMO_J HDL_J TCHOL_J; do
  curl -s -L -o "$D/$f.xpt" "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/$f.xpt"
done
