# Running our Model on a Microcontroller using MicroTVM and ESP-IDF

This directory contains the target software that runs on the ESP32-C3. The application listens to its surroundings with a microphone and indicates when it has detected a word.

## Structure

It is recommended to go through the tutorial in the following order while performing the explained steps:

0. Setup the Toolchain: [`setup_espidf.md`](setup_espidf.md)
1. Follow MicroTVM Tutorial: [`microtvm_tutorial_tvmc.ipynb`](microtvm_tutorial_tvmc.ipynb)

## Prerequisites

The following tasks depend on the `2_deploy/gen/mlf.tar` and `2_deploy/gen/mlf_tuned.tar` files generated in the previous step. Please finish these first, before continuing here.

See `2_deploy/README.md` for **Steps 1.-6.**!

**7. Setup ESP-IDF**

See [`install_tvm.md`](install_tvm.md)!

## Usage

First, start jupyter notebook inside the virtual environment

```bash
jupyter notebook  # Alternative: `jupyter lab`
```

If using a remote host, append: ` --no-browser --ip 0.0.0.0 --port XXXX` (where XXXX should be a number greater than 1000)

## MicroTVM First Steps

## Target SW

you can find all relevant information in the `README.md` files inside the generated `prj` or `prj_tuned` directories respectively inside `microtvm/template_project/src/micro_kws/README.md`

## Tasks

**Lab 2 Tasks (Part 1):**

1. TODO