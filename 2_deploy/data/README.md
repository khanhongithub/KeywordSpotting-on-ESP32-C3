# Provided data for Lab 2 Part 1

## Model Files

- `micro_kws_student_quantized.tflite`: MicroKWS model architecture to be used in Lab 2 (Keywords: **yes,no,up,down,left,right,on,off**)
- `micro_kws_xs_quantized.tflite`: Example model using a very small model architecture. (Keywords: **yes,no**)

## Tuning Logs

Any of the following files can be passed to TVM to generate target optimized operator implementations.

- `micro_kws_student_tuning_log_nchw.txt`: Full AutoTVM tuning log for provided MicroKWS model architecture
- `micro_kws_student_tuning_log_nchw_best.txt`: This file contains only the best tuning results per tunable-tasks

## Test Data

The following files are extracted from the speech_commands dataset using in Lab 1 using the script `wav2features.py`.

- `no.npz`: A compressed numpy array from a preprocessed "no" sample
- `yes.npz`: A compressed numpy array from a preprocessed "yes" sample
