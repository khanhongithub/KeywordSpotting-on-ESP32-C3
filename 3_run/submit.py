import os
import argparse
from pathlib import Path
from zipfile import ZipFile


NUM_WORDS = 8
WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off"]
MODEL_NAME = "micro_kws_student"
QUANTIZED = True


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
    gen = directory / ".." / "2_deploy" / "gen"
    mlf = gen / "mlf.tar"
    assert mlf.is_file(), f"{mlf} file does not exist"
    mlf_tuned = gen / "mlf_tuned.tar"
    assert mlf_tuned.is_file(), f"{mlf_tuned} file does not exist"
    prj = directory / "prj"
    assert prj.is_dir(), f"{prj} directory does not exist"
    prj_tuned = directory / "prj_tuned"
    assert prj_tuned.is_dir(), f"{prj_tuned} directory does not exist"

    # Metadata
    names_path = student / "names.txt"
    assert names_path.is_file(), f"{names_path} file does not exist"
    assert count_names(names_path) > 0
    words_path = student / "words.txt"
    assert words_path.is_file(), f"{words_path} file does not exist"
    words = get_words(words_path)
    assert len(words) == NUM_WORDS, f"Expected {NUM_WORDS} in student/words.txt"
    # assert WORDS == words, "Please use words matching the provided model"

    return [
        names_path,
        words_path,
        mlf,
        mlf_tuned,
    ], [
        prj,
        prj_tuned,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="submission_2.zip")
    args = parser.parse_args()

    directory = Path(".")
    files, directories = gather_files(directory)

    print(f"Creating archive: {args.out}\n")

    with ZipFile(args.out, "w") as handle:
        for file in files:
            print(f"Adding file: {file}")
            handle.write(file)
        for directory in directories:
            for dirname, subdirs, files in os.walk(directory):
                if "/build" in dirname:
                    continue
                for file in files:
                    filename = os.path.join(dirname, file)
                    print(f"Adding file: {filename}")
                    handle.write(filename)
        print(f"\nDone. Please upload the {args.out} file to Moodle!")


if __name__ == "__main__":
    main()
