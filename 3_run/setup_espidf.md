# Install the ESP-IDF

**Note:** Since the ESP-IDF is already installed on the chair computers you can skip this part if you are working on one.

Follow the instructions of the [ESP-IDF get started guide](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html) to set up and install the ESP-IDF.

**Note:** Make sure you are cloning `release/v4.4` and installing the IDF for ESP32-C3 targets `./install.sh esp32c3` as explained below!

In later sections, it is assumed that the `IDF_DIR` environment variable points to the directory where `esp-idf` was cloned. On your local machine you will need to ensure that the `IDF_DIR` points to your ESP-IDF installation yourself. On chair computers this variable is set by the `setup.sh` script.

#### Ubuntu / Debian

If you are using Ubuntu or Debian the steps would be as follows:

1. Install the required dependencies
```
sudo apt-get install git wget flex bison gperf python3 python3-pip python3-setuptools cmake ninja-build ccache libffi-dev libssl-dev dfu-util libusb-1.0-0
```

2. Ensure you have at least Python 3.6, better Python 3.8
```
python3 --version
```

3. Clone the ESP-IDF GitHub repository
```
git clone --recursive --single-branch -b release/v4.4 https://github.com/espressif/esp-idf.git
```
  *Ensure you are cloning the v4.4 repository, as this is the version we are using! Different versions might lead to compilation errors!*

4. Install the ESP-IDF
```
cd esp-idf
export IDF_DIR=$(pwd)
```
and
```
$IDF_DIR/install.sh esp32c3
```
This will install the IDF for ESP32-C3 targets.

5. You can now source the `export.sh` script. This will make the command-line tools available
```
source $IDF_DIR/export.sh
```
  *Note:* `. $IDF_DIR/export.sh` *does the exact same.*

6. You should now test your installation by running
```
idf.py --version
```
If you see something along the lines of `ESP-IDF v4.4.*` (note the `v4.4`) you have successfully installed the ESP-IDF.

**Note:** Whenever you open a new terminal you will need to repeat the first part of step 4., i.e. `export` the `IDF_DIR` variable to point to your `esp-idf` installation, as well as step 5., i.e. source the `export.sh` script!
