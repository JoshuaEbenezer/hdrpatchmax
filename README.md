# HDRPatchMAX: No-Reference Video Quality Assessment for HDR and SDR using contrast-based segmentation

This repository contains code for HDRPatchMAX and HDRMAX.

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


## Testing 

First extract features from the video whose quality you want to measure.

To extract features, run (for eg.)
```

python3 hdrpatchmax.py --input_file I.yuv --results_file O.z --width 3840 --height 2160 --bit_depth 10 --color_space BT2020

```

Then run 
```

python3 test_single_video.py --feature_file O.z

```

This will output a single quality score.
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

# HDRMAX

Only feature extraction is supported for HDRMAX.

## Feature extraction

To extract features, run (for eg.)
```

python3 hdrmax.py --input_file I.yuv --results_file O.z --width 3840 --height 2160 --bit_depth 10 --color_space BT2020

```

HDRMAX features can be combined with features from other VQA algorithms and jointly trained with a Random Forest or SVR to predict VQA for HDR and SDR. HDRMAX features make SDR VQA algorithms robust to bit depth.
