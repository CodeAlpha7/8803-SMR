import sys

class OutputCapture:
    def __init__(self, output_file):
        self.output_file = output_file
        self.original_stdout = sys.stdout
        sys.stdout = self

    def write(self, text):
        self.original_stdout.write(text)
        self.output_file.write(text)
        self.output_file.flush()

    def flush(self):
        self.original_stdout.flush()
        self.output_file.flush()

def redirect_output_to_file_and_stdout(output_filename):
    file = open(output_filename, 'w')
    return OutputCapture(file)

def reset_output(output_capture):
    sys.stdout = output_capture.original_stdout
    output_capture.output_file.close()