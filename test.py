##############################################################
# Note to students:                                          #
# Do NOT modify this file!                                   #
# If you get errors, you need to fix those in *your* code.   #
# Modifying this code may lead you to generate an upload.zip #
# that will not work in our grader.                          #
##############################################################

import os, subprocess, time
import logging
import traceback


class ComparisonError(ValueError):
    pass


def get_num_corrrect(a, b):
    if len(a) != len(b):
        raise ComparisonError("strings to compare must have the same length")
    # Return the number of locations where the two strings are equal
    return sum(int(i == j) for i, j in zip(a, b))


def test(executable_path, plaintext, ciphertext, breakpoint):
    """
    return: elapsed_time, num_correct, output
    """
    if not os.path.exists(executable_path):
        logging.error("decode-cli does not exist")
        raise EnvironmentError("decode-cli does not exist")

    subprocess.call(["chmod", "+x",
                     executable_path])  # Ensure executable can be executed
    executable_file = os.path.basename(
        executable_path)  # foo/bar/decode -> decode
    executable_dir = os.path.dirname(
        executable_path)  # foo/bar/decode -> foo/bar

    start_dir = os.getcwd()
    os.chdir(
        executable_dir
    )  # CHANGE TO CODE DIRECTORY (the student"s code may use relative paths)

    try:
        start_time = time.time()
        output = subprocess.check_output(
            ["./" + executable_file, ciphertext,
             str(breakpoint)],
            stderr=subprocess.STDOUT,
            encoding="UTF-8").strip("\r\n")
        end_time = time.time()
    except subprocess.CalledProcessError as e:
        os.chdir(start_dir)  ###### CHANGE BACK TO ORIGINAL DIRECTORY
        logging.error("decode-cli failed")
        logging.error(e.output)
        raise

    os.chdir(start_dir)  ###### CHANGE BACK TO ORIGINAL DIRECTORY

    elapsed_time = end_time - start_time
    num_correct = get_num_corrrect(plaintext, output)

    return elapsed_time, num_correct, output


def first_line(filename):
    # Return first line of file as string, without trailing newline
    with open(filename) as f:
        return f.readline().rstrip("\r\n")


def main():
    logging.basicConfig(format="%(levelname)s - %(message)s")

    executable = "./decode-cli"
    plaintext = first_line("data/test/plaintext.txt")
    ciphertext = first_line("data/test/ciphertext.txt")
    ciphertext_with_breakpoint = first_line(
        "data/test/ciphertext_breakpoint.txt")
    dummy_text = "the quick brown fox jumped over the lazy dog."

    try:
        print("Running no breakpoint test...")
        elapsed_time, num_correct, _ = test(executable, plaintext, ciphertext,
                                            False)
        print(f"Score (no breakpoint): {num_correct} out of {len(plaintext)}")
        print(f"Elapsed time (no breakpoint): {elapsed_time}")

        print("Running breakpoint test...")
        elapsed_time, num_correct, _ = test(executable, plaintext,
                                            ciphertext_with_breakpoint, True)
        print(f"Score (breakpoint): {num_correct} out of {len(plaintext)}")
        print(f"Elapsed time (breakpoint): {elapsed_time}")

        print("Checking that you are not hardcoding inputs...")
        # make sure you are not hardcoding read input
        test(executable, dummy_text, dummy_text, False)
    except ComparisonError:
        print("!!! ERROR !!!")
        print("Plaintext output is not the same length as the ciphertext.")
        exit(-1)
    except:
        print("!!! ERROR !!!")
        print("Your code seems to have errors.",
              "Please fix them and then rerun this test.")
        exit(-1)

    print("Creating an upload.zip that you can submit...")
    subprocess.call(["zip", "-rq", "upload.zip"] + sorted(os.listdir(".")))


if __name__ == "__main__":
    main()
