# HDR CHIPQA: No-Reference Video Quality Assessment for HDR using Space-Time Chips

This repository contains code for HDR ChipQA and ChipQA.

## Requirements

Create a conda environment from the specification file  hdrpatchmax_spec-file.txt using
```

conda create --name hdrpatchmax --file conda_spec-file.txt

```
Activate the environment. There are some packages only pip can install, so also do 

```

pip install -r pip_requirements.txt

```

You should now be good to go!

## Feature extraction

To extract features, run (for eg.)
```

python3 hdrpatchmax.py --input_file I.yuv --results_file O.z --width 3840 --height 2160 --bit_depth 10 --color_space BT2020

```

## Training 

Run 
```

python3 randomforest.py --score_file score.csv --feature_folder ./folder --train_and_test

```
to evaluate. Other options can be seen with the -h option.


