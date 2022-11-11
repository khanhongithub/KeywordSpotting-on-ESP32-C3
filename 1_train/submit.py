import argparse
from pathlib import Path
from zipfile import ZipFile

NUM_WORDS = 8
MODEL_NAME = "micro_kws_student"


def get_lines(file):
    with open(file, "r") as handle:
        lines = handle.readlines()
    stripped = [line.strip() for line in lines if len(line) > 0]
    return stripped


def count_lines(file):
    return len(get_lines(file))


def count_names(file):
    names = []
    for line in get_lines(file):
        assert "#" not in line, "# is not allowed in names.txt"
        names.append(line)
    return len(names)


def get_words(file):
    lines = get_lines(file)
    assert len(lines) == 1, "words.txt should only have a single line with comma separated values"
    line = lines[0]
    words = line.split(",")
    return words


def gather_files(directory):
    student = directory / "student"
    assert student.is_dir(), f"{student} directory does not exist"
    models = directory / "models"
    assert models.is_dir(), f"{models} directory does not exist"

    # Metadata
    names_path = student / "names.txt"
    assert names_path.is_file(), f"{names_path} file does not exist"
    assert count_names(names_path) > 0
    words_path = student / "words.txt"
    assert words_path.is_file(), f"{words_path} file does not exist"
    words = get_words(words_path)
    assert len(words) == NUM_WORDS, f"Expected {NUM_WORDS} in student/words.txt"

    # Code
    model_code_path = student / "model.py"
    assert model_code_path.is_file(), f"{model_code_path} file does not exist"
    callbacks_code_path = student / "callbacks.py"
    assert callbacks_code_path.is_file(), f"{callbacks_code_path} file does not exist"
    metrics_code_path = student / "metrics.py"
    assert metrics_code_path.is_file(), f"{metrics_code_path} file does not exist"
    estimate_code_path = student / "estimate.py"
    assert estimate_code_path.is_file(), f"{estimate_code_path} file does not exist"

    # Models
    words_str = "".join(words)
    model_path = models / f"{MODEL_NAME}_{words_str}.tflite"
    assert model_path.is_file(), f"{model_path} file does not exist"
    model_quantized_path = models / f"{MODEL_NAME}_{words_str}_quantized.tflite"
    assert model_quantized_path.is_file(), f"{model_quantized_path} file does not exist"

    # Accuracy
    # accuracy_path = directory / "accuracy.txt"
    # assert accuracy_path.is_file(), f"{accuracy_path} file does not exist"

    # Estimations
    # estimations_path = directory / "estimations.txt"
    # assert estimations_path.is_file(), f"{estimations_path} file does not exist"

    return [
        names_path,
        words_path,
        model_code_path,
        callbacks_code_path,
        metrics_code_path,
        estimate_code_path,
        model_path,
        model_quantized_path,
        # accuracy_path,
        # estimations_path,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="submission_1.zip")
    args = parser.parse_args()

    directory = Path(".")
    files = gather_files(directory)

    print(f"Creating archive: {args.out}\n")

    with ZipFile(args.out, "w") as handle:
        for file in files:
            print(f"Adding file: {file}")
            handle.write(file)
        print(f"\nDone. Please upload the {args.out} file to Moodle!")


if __name__ == "__main__":
    main()
