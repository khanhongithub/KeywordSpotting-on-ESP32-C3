VENV_DIR=venv
IDF_DIR=../../../common/tools/toolchain/esp-idf/
IDF_TOOLS_DIR=../../../common/tools/toolchain/espressif/
# WORDS=yesnoupdownleftrightonoff
# MODEL_NAME=micro_kws_student_$(WORDS)_quantized
MODEL_NAME=micro_kws_student_quantized
LAYOUT=NCHW
PORT=/dev/ttyUSB0

ifeq (, $(shell which python3.8))
  $(error "Could not find Python 3.8 executable!")
endif

help:
	@echo "Help: Run 'make clean && make submission' to run basic tests and generate the submission.zip for the moodle upload."

venv: requirements.txt
	@echo "Setting up Python virtual environment (This may take a very long time)"
	@test -d $(VENV_DIR) || virtualenv -p python3.8 $(VENV_DIR) || python3.8 -m venv $(VENV_DIR)
	@. $(VENV_DIR)/bin/activate && pip install -r requirements.txt > /dev/null

install: | venv
	@. $(VENV_DIR)/bin/activate && pip install tlcpack-nightly -f https://tlcpack.ai/wheels > /dev/null

lab: | install
	 . $(VENV_DIR)/bin/activate && jupyter lab --ip 0.0.0.0

gen/module.tar: | install
	. $(VENV_DIR)/bin/activate && tvmc compile data/$(MODEL_NAME).tflite --output gen/module.tar --target llvm --desired-layout $(LAYOUT)

gen/module: gen/module.tar | install
	@mkdir -p gen/module && tar xf gen/module.tar -C gen/module

gen/module_tuned.tar: | install
	. $(VENV_DIR)/bin/activate && tvmc compile data/$(MODEL_NAME).tflite --output gen/module_tuned.tar --target llvm --desired-layout $(LAYOUT) --tuning-records data/$(MODEL_NAME)_tuning_log_$(LAYOUT)_best.txt

gen/module_tuned: gen/module_tuned.tar | install
	@mkdir -p gen/module_tuned && tar xf gen/module_tuned.tar -C gen/module_tuned

student/out.txt: gen/module.tar | install
	. $(VENV_DIR)/bin/activate && tvmc run gen/module.tar --fill-mode random --print-time --print-top 10 > student/out.txt

student/out_tuned.txt: gen/module_tuned.tar | install
	. $(VENV_DIR)/bin/activate && tvmc run gen/module_tuned.tar --fill-mode random --print-time --print-top 10 > student/out_tuned.txt

gen/mlf.tar: | install
	. $(VENV_DIR)/bin/activate && tvmc compile data/$(MODEL_NAME).tflite --output gen/mlf.tar --target c --runtime crt --runtime-crt-system-lib 1 --desired-layout $(LAYOUT) --output-format mlf --pass-config tir.disable_vectorize=1

gen/mlf: gen/mlf.tar | install
	@mkdir -p gen/mlf && tar xf gen/mlf.tar -C gen/mlf/

gen/mlf_tuned.tar: | install
	. $(VENV_DIR)/bin/activate && tvmc compile data/$(MODEL_NAME).tflite --output gen/mlf_tuned.tar --target c --runtime crt --runtime-crt-system-lib 1 --desired-layout $(LAYOUT) --output-format mlf --pass-config tir.disable_vectorize=1 --tuning-records data/$(MODEL_NAME)_tuning_log_$(LAYOUT)_best.txt

gen/mlf_tuned: gen/mlf_tuned.tar | install
	@mkdir -p gen/mlf_tuned && tar xf gen/mlf_tuned.tar -C gen/mlf_tuned/

module: gen/module
module_tuned: gen/module_tuned
test: student/out.txt
test_tuned: student/out_tuned.txt
mlf: gen/mlf
mlf_tuned: gen/mlf_tuned

all: module module_tuned mlf mlf_tuned submission

clean:
	rm -rf gen/module.tar
	rm -rf gen/module/
	rm -rf gen/module_tuned.tar
	rm -rf gen/module_tuned/
	rm -rf gen/mlf.tar
	rm -rf gen/mlf/
	rm -rf gen/mlf_tuned.tar
	rm -rf gen/mlf_tuned/

cleanall: clean
	rm -rf venv/

.PHONY: help clean cleanall submission all module module_tuned mlf mlf_tuned install venv lab
