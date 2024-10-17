# mac-batch-screenshot-renamer-ai

## Introduction

mac-batch-screenshot-renamer-ai is a Python class designed to automate the process of renaming screenshot files using AI-generated titles. It leverages OpenAI's GPT models to analyze the content of screenshots and generate informative filenames. Additionally, it offers functionality to group related screenshots based on their timestamps.

## Quick Start

Use the ScreenshotRenamer class in your Python script:
```python
from screenshot_renamer import ScreenshotRenamer

# Initialize the renamer
renamer = ScreenshotRenamer(directory="/path/to/screenshots")

# Rename screenshots
renamer.rename_screenshots()

# Group existing files
renamer.group_existing_files()

# Write grouped files to JSON
renamer.write_grouped_files_to_json()
```

## Initialization Parameters

The `ScreenshotRenamer` class accepts several parameters during initialization:

- `max_filename_length` (int, default: 60): Maximum length of the new filename.
- `model` (str, default: "gpt-4o-mini"): OpenAI model to use for generating titles.
- `openai_key` (Optional[str], default: None): OpenAI API key. If None, it will be read from the environment variable.
- `prompt` (Optional[str], default: None): Custom prompt for the AI. If None, a default prompt will be used.
- `group_files` (bool, default: False): Whether to group files based on timestamp.
- `group_time_threshold` (int, default: 15): Time threshold (in minutes) for grouping files.
- `directory` (Optional[str], default: None): Directory to use for input and output operations. If None, it defaults to "~/Documents".


## Main Methods

### 1. rename_screenshots()

This method renames screenshots in the specified directory using AI-generated titles.

Usage:
```python
renamer.rename_screenshots()
```

Process:
1. Scans the specified directory for screenshot files matching the predefined pattern.
2. For each screenshot:
   - Extracts the creation timestamp from the filename.
   - Sends the image to the OpenAI API for content analysis.
   - Generates a new filename based on the AI's response, adhering to the `max_filename_length`.
   - Sanitizes the new filename to ensure it's valid for the file system.
   - Renames the file with the new name while preserving the original timestamp.
   - If `group_files` is enabled, adds the file to a group based on its timestamp.
3. Handles potential API errors and file system issues gracefully.
4. Provides console output for each renamed file.

### 2. group_existing_files()

This method groups existing renamed files in the specified directory based on their timestamps.

Usage:
```python
renamer.group_existing_files()
```

Process:
1. Scans the specified directory for PNG files that match the renamed format.
2. For each file:
   - Extracts the timestamp from the filename.
   - Converts the timestamp to a datetime object.
3. Groups files based on the extracted timestamps and the `group_time_threshold`:
   - Files within the threshold are placed in the same group.
   - Each group is identified by the earliest timestamp in the group.
4. Merges close groups to prevent excessive fragmentation:
   - If the time difference between the last file of one group and the first file of the next group is within the threshold, the groups are merged.
5. Stores the grouped files information in the `grouped_files` attribute.

### 3. write_grouped_files_to_json()

This method writes the grouped files information to a JSON file in the specified directory.

Usage:
```python
renamer.write_grouped_files_to_json()
```

Process:
1. Checks if `grouped_files` contains data. If not, it calls `group_existing_files()` first.
2. Generates a unique filename for the JSON output:
   - Format: `grouped_screenshots_YYYYMMDD_HHMMSS.json`
   - Uses the current date and time to ensure uniqueness.
3. Creates a dictionary with group timestamps as keys and lists of filenames as values.
4. Writes the dictionary to the JSON file in the specified directory:
   - Uses `json.dump()` with indentation for readability.
5. Provides console output confirming the JSON file creation and its location.

Note: These methods work together to provide a comprehensive solution for renaming, grouping, and documenting screenshots. The `rename_screenshots()` method can be used independently, while `group_existing_files()` and `write_grouped_files_to_json()` are often used in sequence to process already renamed files.
