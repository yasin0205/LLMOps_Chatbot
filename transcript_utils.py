# Path gives a clean, cross-platform way to work with files and folders
from pathlib import Path
# re is Python's regular expression module, used to recognise timestamp lines
import re

# Compile the regex once at module level so it isn't rebuilt for every line.
# It matches a full line shaped like:  00:00:01.000 --> 00:00:03.500
#   ^                 -> start of the line
#   \d{2}:\d{2}:\d{2} -> hours:minutes:seconds, two digits each
#   \.\d{3}           -> a dot followed by three digits (milliseconds)
#   \s+-->\s+         -> the arrow, with one or more spaces around it
#   (same pattern again for the end time)
#   $                 -> end of the line
TIMESTAMP_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}$")


def clean_transcript_text(transcript: str) -> str:
    """Remove timestamp lines from a transcript and return only the spoken text."""
    # Split the whole text into a list of lines (newline characters are removed)
    lines = transcript.splitlines()
    # This list will collect only the lines we want to keep
    cleaned_lines = []

    for line in lines:
        # Remove spaces/tabs at the start and end of the line
        stripped = line.strip()
        if not stripped:  # skip empty lines
            continue
        # If the line looks like "00:00:01.000 --> 00:00:03.500", skip it
        if TIMESTAMP_PATTERN.match(stripped):
            continue
        # Otherwise it is spoken text, so keep it
        cleaned_lines.append(stripped)

    # Join the kept lines back into one string, one line per row
    return "\n".join(cleaned_lines)


def load_and_clean_transcripts(original_dir: Path | str,
                               new_dir: Path | str):
    """Load transcript files from the original directory, clean them, and save to the new directory."""

    ORIGINAL_PATH = Path(original_dir)  # if path is a string, convert to Path
    NEW_PATH = Path(new_dir)

    # Create the new directory if it doesn't exist
    # parents=True -> also create any missing parent folders
    # exist_ok=True -> don't raise an error if the folder already exists
    NEW_PATH.mkdir(parents=True, exist_ok=True)

    # Find all .txt files directly inside the original folder (not in subfolders)
    # glob returns a generator, so files are found as the loop runs
    transcripts = ORIGINAL_PATH.glob("*.txt") 
    for transcript_path in transcripts:
        # read the content of the transcript file
        file_content = transcript_path.read_text(encoding='utf-8')
        # clean the transcript text (remove timestamps and empty lines)
        cleaned_content = clean_transcript_text(file_content)
        # collect file name for new file (e.g. "video1.txt")
        filename = transcript_path.name
        # create output path for the cleaned transcript (same name, new folder)
        output_path = NEW_PATH / filename
        # write the cleaned content to the new file
        output_path.write_text(cleaned_content, encoding='utf-8')


# This block runs only when you execute this file directly (python file.py),
# not when another file imports it
if __name__ == "__main__":

    # Read from ./transcripts and save the cleaned files into ./cleaned-transcripts
    load_and_clean_transcripts(Path("./transcripts"), Path("./cleaned-transcripts"))