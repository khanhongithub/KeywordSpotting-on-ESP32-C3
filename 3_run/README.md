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

See [`setup_espidf.md`](setup_espidf.md)!

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

Complete the tasks in `2_deploy` before continuing here!

**Lab 2 Tasks (Part 2):**

0. Fill out the `3_run/student/names.txt` file appropriately. `3_run/student/words.txt` is already filled out!
1. Read and follow MicroTVM (ESP-IDF) tutorial mentioned above.
2. Generate two project directories using the MicroTVM command-line interface (`tvmc micro create ...`)
    - Using the untuned (`mlf.tar`) TVM artifacts: `3_run/prj/`
    - Using the tuned (`mlf_tuned.tar`) TVM artifacts: `3_run/prj_tuned`
3. Perform the following tasks for both of the MicroTVM projects:
    - Update the project settings to use the new model architecture (e.g. number of keywords, label names etc., See `2_deploy/data/README.md`)
    - Compile, flash and run the demo program. How does it perform?
    - Determine the static ROM and RAM usage of this program using the `idf.py size` command. Can you explain why the software size exceeds the estimated metrics of the model?

5. For the following, only consider the tuned variant.

6. Solve programming challenges:
    - Implement the post-processing introduced in Section 2.2.1 for the MicroKWS software by editing the files

        `3_run/prj_tuned/src/components/student/posterior.cc` & `3_run/prj_tuned/src/components/student/include/posterior.h`
    - Use the `idf.py menuconfig` command to modify the algorithms parameters (history lengths, detection threshold, deactivation period) which should **not** be hardcoded in the C++ file.
7. Create the ZIP for you your Moodle submission using `python submit.py`
