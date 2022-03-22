## MicroKWS Training Flow

### Prerequisites

**1. Clone repository**

```bash
git clone git@gitlab.lrz.de:de-tum-ei-eda-esl/micro-kws.git
```

**2. Enter directory**

```bash
cd micro-kws/1_train
```

**3. Create virtual python environment**

```bash
virtualenv -p python3.8 venv
```

**4. Enter virtual python environment**

```bash
source venv/bin/activate
```

**5. Install python packages into environment**

```bash
pip install -r requirements.txt
```

*Warning:* This command might take several minutes to execute.

### Usage

#### Jupyter Notebook (Use this for lab 1)

First, start jupyter notebook inside the virtual environment

```bash
jupyter notebook MicroKWS.ipynb  # Alternative: `jupyter lab`
```

If using a remote host, append: ` --no-browser --ip 0.0.0.0 --port XXXX` (where XXXX should be a number greater than 1000)

If you experience warnings it might help to use ` --NotebookApp.iopub_msg_rate_limit=1.0e10  --NotebookApp.iopub_data_rate_limit=1.0e10`

**Lab 1 Tasks:**

1. Follow the steps explained in `MicroKWS.ipynb`
2. Solve the tasks introduced in the lab 1 manual in the following files
  - `student/names.txt`: Comma-separated list of students in your group.
  - `student/words.txt`: Comma-separated list of keywords assigned to your group (see Moodle).
  - `student/callbacks.py`: Implementation of "early-stopping" callback for training procedure.
  - `student/metrics.py`: Implementation of methods to calculate per-class recall, precision and f1-score for a given confusion matrix.
  - `student/estimate.py`: Implementation of methods to estimate the ROM/RAM/MACs of a given TFLite model.
  - `student/model.py`: Designed Keras model architecture for programming challenge.
  - `models/micro_kws_student_{your_keywords}.tflite`, `models/micro_kws_student_{your_keywords}_quantized.tflite`: You quantized and unquantized model for the final programming. challenge
3. Update `MicroKWS.ipynb`to use your model (`micro_kws_student`) and keywords instead of the example (`yes,no`) ones. (See `FLAGS.model_name` and `FLAGS.wanted_words`)
4. Run `MicroKWS.ipynb` improve your model architecture until your model fulfils the challenge requirements.
5. Optional: Check your code using provided unit tests (only partially): `python -m pytest tests/`
6. Create a ZIP file for the final submission: `python submit.py`
7. Upload ZIP file to Moodle

#### Command line scripts (Optional)

In addition to some utilities used in the notebook, the flow can also be used on the command line. Here are a few examples:

```bash
python train.py --model_architecture micro_kws_xs
python test.py --model_architecture micro_kws_xs --checkpoint <checkpoint_path>
python convert.py --model_architecture micro_kws_xs --checkpoint <checkpoint_path> --inference_type int8
python test_tflite.py --tflite_path micro_kws_xs.tflite
```

To learn about the available options add `--help` to the previous commands.

**Warning:** The provided scrript may not be in sync with the code you will find in the Jupyter notebook. Please stick to the notebook when solving the lab exercises!

### Disclaimer

This tutorial is inpired by the contents of: https://github.com/ARM-software/ML-examples/tree/main/tflu-kws-cortex-m/Training
