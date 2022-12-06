# Pre-compiled Python packages for TVM

The easiest way to get started using TVM is the TLCPack python package. It is a community maintained binary build, including the TVM deep learning compiler suite. See https://tlcpack.ai/ for more details.

While nightly builds can be installed via `pip install tlcpack-nightly -f https://tlcpack.ai/wheels`, we highly **highly** recommend using a fixed-version build provided by us to work on the lab exercises. For this purpose, we provide the following pre-build binaries, which are hosted in this our [Sync&Share](https://syncandshare.lrz.de/getlink/fi8iHVDmnCY9DSPntsdZQV/tlcpack):

| Operating System | Python Version | Filename                                                                                                                                                                                                                                                               |
|------------------|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Debian/Ubuntu    | Python 3.7     | [`tlcpack_nightly-0.11.dev404+gedfeba5c3-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`](https://syncandshare.lrz.de/dl/fi8iHVDmnCY9DSPntsdZQV/tlcpack/tlcpack_nightly-0.11.dev404+gedfeba5c3-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl)   |
| Debian/Ubuntu    | **Python 3.8** | [`tlcpack_nightly-0.11.dev404+gedfeba5c3-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`](https://syncandshare.lrz.de/dl/fi8iHVDmnCY9DSPntsdZQV/tlcpack/tlcpack_nightly-0.11.dev404+gedfeba5c3-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl)     |
| Debian/Ubuntu    | Python 3.9     | [`tlcpack_nightly-0.11.dev404+gedfeba5c3-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`](https://syncandshare.lrz.de/dl/fi8iHVDmnCY9DSPntsdZQV/tlcpack/tlcpack_nightly-0.11.dev404+gedfeba5c3-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl)     |
| Debian/Ubuntu    | Python 3.10    | [`tlcpack_nightly-0.11.dev404+gedfeba5c3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`](https://syncandshare.lrz.de/dl/fi8iHVDmnCY9DSPntsdZQV/tlcpack/tlcpack_nightly-0.11.dev404+gedfeba5c3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl) |
| MacOS            | Python 3.7     | [`tlcpack_nightly-0.11.dev404+gedfeba5c3-cp37-cp37m-macosx_10_15_x86_64.whl`](https://syncandshare.lrz.de/dl/fi8iHVDmnCY9DSPntsdZQV/tlcpack/tlcpack_nightly-0.11.dev404+gedfeba5c3-cp37-cp37m-macosx_10_15_x86_64.whl)                                                 |
| MacOS            | Python 3.8     | [`tlcpack_nightly-0.11.dev404+gedfeba5c3-cp38-cp38-macosx_10_15_x86_64.whl`](https://syncandshare.lrz.de/dl/fi8iHVDmnCY9DSPntsdZQV/tlcpack/tlcpack_nightly-0.11.dev404+gedfeba5c3-cp38-cp38-macosx_10_15_x86_64.whl)                                                   |
| Windows          | Python 3.7     | [`tlcpack_nightly-0.11.dev404+gedfeba5c3-cp310-cp310-win_amd64.whl`](https://syncandshare.lrz.de/dl/fi8iHVDmnCY9DSPntsdZQV/tlcpack/tlcpack_nightly-0.11.dev404+gedfeba5c3-cp310-cp310-win_amd64.whl)                                                                   |

These files are based on the commit `d6632070a01e23270f9f480efc39d09fc38eb55f` in the upstream TVM repository from Nov 21, 2022.

Download the right one for your system and execute `pip install tlcpack_nightly-*.whl` to install it into your virtual environment.

Verify your TVM installation by running `tvmc --version`.
