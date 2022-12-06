# MicroTVM ESP-IDF Template

Project API files to deploy MicroTVM Models to ESP-IDF compatible devices (ESP32 & ESP32-C3)

## Usage

Example usage:

```bash
tvmc micro create ./prj mlf.tar template --template-dir /path/to/template_project --project-option project_type=host_driven idf_target=esp32
tvmc micro build -./prj template --template-dir /path/to/template_project --project-option verbose=false
tvmc micro flash ./prj template --template-dir /path/to/template_project
```

## Prerequisites

First, make sure to have an up-to-date version of TVM installed, either by
- Installing a nightly build from TLCPack (See https://tlcpack.ai)
- Compiling it from soruce (See https://tvm.apache.org/docs/install/from_source.html)

Additionally a supported version (v4.4) of the ESP-IDF Toolchain should be installed manually. Please follow the official documentation:
- Windows (untested): https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/windows-setup.html
- Linux and macOS: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/linux-macos-setup.html

## Configuration

The Project API templates supports the following config options:

- `extra_files_tar`: If given, during generate_project, uncompress the tarball at this path into the project dir.
- `project_type`: Type of project to generate. (Choices: `host_driven`, `micro_kws`)
- `verbose`: Run build with verbose output.
- `idf_target`: Name of the Espressif board to build for. (Choices: `esp32`, `esp32c3`)
- `idf_serial_port`: Name of the serial port. (Leave empty to automatic lookup)
- `idf_serial_baud`: Baudrate of the serial port.
- `warning_as_error`: Treat warnings as errors and raise an Exception.

## Compatibility

The codebase has been tested on the following environments:

- Operating Systems: MacOS 11.7, Ubuntu 20.04
- Python Versions: v3.8
- TVM Version: v0.11
- ESP-IDF Version: v4.4