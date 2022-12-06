VENV_DIR=venv
IDF_DIR=$(LAB_TOOLS_DIR)/toolchain/esp-idf/
IDF_TOOLS_DIR=$(LAB_TOOLS_DIR)/toolchain/espressif/
GEN_DIR=../2_deploy/gen/
LAYOUT=NCHW
PORT=/dev/ttyUSB0
TARGET=esp32c3
PROJECT_TYPE=micro_kws
VERBOSE=false

ifeq (, $(shell which python3.8))
  $(error "Could not find Python 3.8 executable!")
endif

help:
	@echo "Help: Run 'make clean && make submission' to run basic tests and generate the submission_2.zip for the moodle upload."

venv: requirements.txt
	@echo "Setting up Python virtual environment (This may take a very long time)"
	@test -d $(VENV_DIR) || virtualenv -p python3.8 $(VENV_DIR) || python3.8 -m venv $(VENV_DIR)
	@. $(VENV_DIR)/bin/activate && pip install -r requirements.txt > /dev/null

install: | venv
	@. $(VENV_DIR)/bin/activate && pip install tlcpack-nightly -f https://tlcpack.ai/wheels > /dev/null

lab: | install
	 . $(VENV_DIR)/bin/activate && jupyter lab --ip 0.0.0.0

prj: $(GEN_DIR)/mlf.tar | install
	echo X=$(LAB_TOOLS_DIR)
	export IDF_TOOLS_PATH=$(IDF_TOOLS_DIR) && . $(VENV_DIR)/bin/activate && . $(IDF_DIR)/export.sh && tvmc micro create -f ./prj $(GEN_DIR)/mlf.tar template --template-dir microtvm/template_project --project-option project_type=$(PROJECT_TYPE) idf_target=$(TARGET)

prj_tuned: $(GEN_DIR)/mlf_tuned.tar | install
	export IDF_TOOLS_PATH=$(IDF_TOOLS_DIR) && . $(VENV_DIR)/bin/activate && . $(IDF_DIR)/export.sh && tvmc micro create -f ./prj_tuned $(GEN_DIR)/mlf_tuned.tar template --template-dir microtvm/template_project --project-option project_type=$(PROJECT_TYPE) idf_target=$(TARGET)

prj/build: prj | install
	export IDF_TOOLS_PATH=$(IDF_TOOLS_DIR) && . $(VENV_DIR)/bin/activate && . $(IDF_DIR)/export.sh && tvmc micro build -f ./prj template --template-dir microtvm/template_project --project-option verbose=false idf_target=$(TARGET)

prj_tuned/build: prj_tuned | install
	export IDF_TOOLS_PATH=$(IDF_TOOLS_DIR) && . $(VENV_DIR)/bin/activate && . $(IDF_DIR)/export.sh && tvmc micro build -f ./prj_tuned template --template-dir microtvm/template_project --project-option verbose=$(VERBOSE) idf_target=$(TARGET)

flash: | compile
	export IDF_TOOLS_PATH=$(IDF_TOOLS_DIR) && . $(VENV_DIR)/bin/activate && . $(IDF_DIR)/export.sh && tvmc micro flash ./prj template --template-dir microtvm/template_project --project-option idf_target=$(TARGET)

flash_tuned: | compile_tuned
	export IDF_TOOLS_PATH=$(IDF_TOOLS_DIR) && . $(VENV_DIR)/bin/activate && . $(IDF_DIR)/export.sh && tvmc micro flash ./prj_tuned template --template-dir microtvm/template_project --project-option idf_target=$(TARGET)

student/submission_2.zip: prj prj_tuned student/words.txt student/names.txt | install
	@echo "Generating submission files"
	. $(VENV_DIR)/bin/activate && python submit.py --out submission_2.zip

prj/test/build: prj/test
	export IDF_TOOLS_PATH=$(IDF_TOOLS_DIR) && . $(VENV_DIR)/bin/activate && . $(IDF_DIR)/export.sh && cd prj/test/ && idf.py set-target $(TARGET)

test: prj/test/build
	export IDF_TOOLS_PATH=$(IDF_TOOLS_DIR) && . $(VENV_DIR)/bin/activate && . $(IDF_DIR)/export.sh && cd prj/test/ && idf.py flash monitor

prj_tuned/test/build: prj_tuned/test
	export IDF_TOOLS_PATH=$(IDF_TOOLS_DIR) && . $(VENV_DIR)/bin/activate && . $(IDF_DIR)/export.sh && cd prj_tuned/test/ && idf.py set-target $(TARGET)

test_tuned: prj_tuned/test/build
	export IDF_TOOLS_PATH=$(IDF_TOOLS_DIR) && . $(VENV_DIR)/bin/activate && . $(IDF_DIR)/export.sh && cd prj_tuned/test/ && idf.py flash monitor

project: prj
project_tuned: prj_tuned
compile: prj/build
compile_tuned: prj_tuned/build
submission: student/submission_2.zip

all: project project_tuned compile compile_tuned profile profile_tuned submission test_tuned

clean:
	rm -f submission_2.zip
	rm -rf prj/build
	rm -rf prj/test/build
	rm -rf prj_tuned/build
	rm -rf prj_tuned/test/build

cleanall: clean
	rm -rf venv/
	rm -rf prj/
	rm -rf prj_tuned/


.PHONY: help clean cleanall submission all project project_tuned compile compile_tuned flash flash_tuned venv test test_tuned install lab
