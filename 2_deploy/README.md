# Deploying a MicroKWS Model using TVM

This directory contains various tutorials to generate and test kernels for the MicroKWS model using the TVM ML compiler framework.

## Structure

It is recommended to go through the tutorial in the following order while performing the explained steps:

0. Short guide on how to install TVM on your system: [`install_tvm.md`](install_tvm.md)
1. TVMC Command Line Tutorial: [`tutorial_tvmc.md`](tutorial_tvmc.ipynb)
2. Model Library Format Overview: [`mlf_overview.md`](mlf_overview.md)
3. *Optional:* TVM Python API Tutorial: [`tutorial_python.ipynb`](tutorial_python.ipynb)


## Prerequisites

**1. Clone repository**

```bash
git clone git@gitlab.lrz.de:de-tum-ei-eda-esl/micro-kws.git
```

**2. Enter directory**

```bash
cd micro-kws/2_deploy
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

**6. Setup your TVM installation**

See [`install_tvm.md`](install_tvm.md)!

## Usage

First, start jupyter notebook inside the virtual environment

```bash
jupyter notebook  # Alternative: `jupyter lab`
```

If using a remote host, append: ` --no-browser --ip 0.0.0.0 --port XXXX` (where XXXX should be a number greater than 1000)

## Tasks
**Lab 2 Tasks (Part 1):**

1. TODO

## Useful resources

- TVM Documentation: https://tvm.apache.org/docs/
- TVM Repository: https://github.com/apache/tvm
- MicroTVM: https://tvm.apache.org/docs/topic/microtvm/index.html
