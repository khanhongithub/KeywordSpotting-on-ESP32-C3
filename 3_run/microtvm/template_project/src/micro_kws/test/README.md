# Unit test app for the MicroKWS project

## Test Implementations

The implementation of the test for a specific component (such as `components/student/`) are defined in the components `test` directory (i.e. `components/student/test/`). This top-level `test` diretcory contains the actual application required to run the unit test on a MCU device!

For more information on the provided test cases, please refer to `components/student/test/README.md` instead!

## Setup

### On-device testing

To run these unit test a target binary need to be compiled and flashed to a real MCU device. The result of the test can be inspected via the ESP-IDF serial monitor (`idf.py monitor`).

*How to run the MicroKWS tests on the ESP32-C3 MCU?*

1. Go to the `test` directory of the actual MicroKWS project (`test/` **not** `components/student/test`).
2. Setup an ESP-IDF project as usual: `idf.py set-target esp32c3`
3. Compile the test app: `idf.py build`
4. Flash the app to the device: `idf.py flash`
5. Use the IDF monitor to check the test outputs: `idf.py monitor`

### Testing on Linux host

Currently NOT supported.

## Troubleshooting

If you have the feeling that the test cases du not match the description of the algorithm in the lab manual and the C docstrings/comments, please reach out to the lab team ASAP. We tried to add tolerances to some checks to give some degree to freedom for your implementations.
