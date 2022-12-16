# MicroKWS Target Software

## Getting started

### Build the Code

Don't forget to run `. $IDF_DIR/export.sh` once per terminal session before using the `idf.py` tool.

Change into the `target/` directory (where the `sdkconfig.defaults` is located).

Set the chip target to ESP32-C3

```
idf.py set-target esp32c3
```

Then build with `idf.py`
```
idf.py build
```

### Upload and Run the Code

Don't forget to run `. $IDF_DIR/export.sh` once per terminal session before using the `idf.py` tool.

To flash/upload simply run
```
idf.py --port /dev/ttyUSB0 flash
```
**Note:** You might need to change the serial port of the device connected to your computer. You can list connected devices with `ls /dev/ttyUSB*` or `ls /dev/ttyACM*`. If you have more than one device, simply unplug the ESP32-C3 USB cable and check which device disappears. You might even be able to leave out the `--port` argument and only run `idf.py flash`. This way, the ESP-IDF will try to guess which device is the ESP32-C3. However, your mileage with this feature may vary...

As soon as the upload is finished, the ESP32-C3 will reset and the code will start running!

To monitor the serial output of the ESP32-C3 coming back via UART, run
```
idf.py --port /dev/ttyUSB0 monitor
```

Use `CTRL+]` to exit (this is not a `J`, but a "closing bracket" `]`).

The previous two commands can also be combined
```
idf.py --port /dev/ttyUSB0 flash monitor
```
**Note:** `flash` will also run `build` before flashing. So no need to explicitly `build` before `flash`ing.

## Configuration of MicroKWS

The ESP-IDF uses so-called "KConfig" files to define options to easily change various features without modifying any code. It is documented here: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/kconfig.html. The "GUI", which can be accessed via the `idf.py menuconfig` command, is text-based but should be straightforward given the shortcuts shown at the bottom of the window. Just make sure to properly save your changes to the `sdkconfig` file in the `target` directory (it will ask you whether you would like to save before you exit the GUI). After changing the project configuration, the program will be compiled from scratch.

We added a `MicroKWS Options` submenu to the `menuconfig` interface where you can change various things, such as:
- Modify model hyperparameters (These have to match the ones used during training)
- Configuration of the posterior handler
- Definition of used labels
- Debugging-related toggles, i.e. reduce or increase verbosity

## Debugging MicroKWS

As an alternative to `printf`-based debugging, which is often cumbersome for data-driven programs, we provide an interactive debugging tool to visualize the model's inputs and outputs. A detailed explanation can be found in the `4_debug` directory at the top level of this repository.

To debug the static memory usage of a program, the ESP-IDF provides several commands that are explained in the official documentation (https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/performance/size.html):
- `idf.py size`
- `idf.py size-components`
- `idf.py size-files`

## Programming Challenge

The sources for the posterior handling which needs to be implemented can be found in the `student` ESP-IDF component: [`components/student`](components/student). More information can be found in [`components/student/README.md`](components/student/README.md) and docstrings. How to use the provided unit-tests is explained in [`components/student/test/README.md`](components/student/test/README.md).
