import os
import glob
from datetime import datetime, timedelta
import json
from openai import OpenAI
import base64
import re
from typing import Optional, Dict, List, Tuple
import shutil  # Add this import at the top of the file

class ScreenshotRenamer:
    """
    A class to rename screenshot files using AI-generated titles.
    """

    def __init__(self, max_filename_length: int = 60, model: str = "gpt-4o-mini",
                 openai_key: Optional[str] = None, prompt: Optional[str] = None,
                 group_files: bool = False, group_time_threshold: int = 15,
                 directory: Optional[str] = None) -> None:
        """
        Initialize the ScreenshotRenamer.

        :param max_filename_length: Maximum length of the new filename.
        :param model: OpenAI model to use for generating titles.
        :param openai_key: OpenAI API key. If None, it will be read from environment variable.
        :param prompt: Custom prompt for the AI. If None, a default prompt will be used.
        :param group_files: Whether to group files based on timestamp.
        :param group_time_threshold: Time threshold (in minutes) for grouping files.
        :param directory: Directory to use for input and output operations.
        """
        print(f"Initializing ScreenshotRenamer with max_filename_length={max_filename_length}, model={model}")
        self.screenshot_pattern: str = "Screenshot ????-*"
        self.max_filename_length: int = max_filename_length
        if openai_key is None:
            openai_key = os.getenv("OPENAI_API_KEY")
            print(f"Using OpenAI API key from environment variable: {openai_key[:5]}...{openai_key[-5:]}")
        else:
            print("Using provided OpenAI API key")
        self.client: OpenAI = OpenAI(api_key=openai_key)
        self.model: str = model
        if prompt is None:
            self.prompt: str = f"Look at the content of this screenshot and give a title for it that I can use as an informative filename. It can be at most {self.max_filename_length} characters long. If the screenshot looks like a Zoom session, you can ignore any images of participants and chat. Note that if it is a slide it may include the logo of the slide creator - so you can incorporate that logo text into the new filename. Only return the filename, no pre-amble or post-comments."
        else:
            self.prompt = prompt
        self.group_files: bool = group_files
        self.group_time_threshold: int = group_time_threshold
        self.grouped_files: Dict[str, List[str]] = {}
        self.creation_time = datetime.now()  # Store the creation time
        self.directory = directory or os.path.expanduser("~/Documents")
        print(f"Working directory set to: {self.directory}")

    def rename_screenshots(self) -> None:
        """
        Rename screenshots in the specified directory.
        """
        print(f"Starting rename_screenshots process in {self.directory}")
        self._process_directory(self.directory)
        self._merge_close_groups()

    def _process_directory(self, directory: str) -> None:
        """
        Process a directory to find and rename screenshot files.

        :param directory: Path to the directory to process.
        """
        screenshot_files: List[str] = glob.glob(os.path.join(directory, f"{self.screenshot_pattern}.png"))
        print(f"Found {len(screenshot_files)} screenshot(s) in {directory}")

        for screenshot in screenshot_files:
            if os.path.basename(screenshot).startswith("Screenshot"):
                print(f"Processing screenshot: {screenshot}")
                new_name: Optional[str] = self._get_new_filename(screenshot)
                if new_name:
                    self._rename_file(screenshot, new_name)
                else:
                    print(f"Failed to get new filename for {screenshot}")
            else:
                print(f"Ignoring non-screenshot file: {screenshot}")

    def _get_new_filename(self, filepath: str) -> Optional[str]:
        """
        Get a new filename for the screenshot using AI.

        :param filepath: Path to the screenshot file.
        :return: New filename or None if failed.
        """
        print(f"Getting new filename for: {filepath}")
        with open(filepath, "rb") as f:
            image_data: bytes = f.read()
        print(f"Read {len(image_data)} bytes from {filepath}")
        image_data = base64.b64encode(image_data)
        print(f"Sending request to OpenAI API with model: {self.model}")
        response = self.client.chat.completions.create(
        model=self.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url":
                            {"url":
                            "data:image/jpeg;base64," + image_data.decode("utf-8"),
                            "detail": "high"
                            },
                    }
                ]
            }
        ])

        reply: str = response.choices[0].message.content
        print(f"Received response from OpenAI: {reply}")
        return reply

    def _rename_file(self, old_path: str, new_name: str) -> None:
        """
        Rename a file and optionally add it to a group.

        :param old_path: Current path of the file.
        :param new_name: New name for the file.
        """
        print(f"Renaming file: {old_path}")
        extension: str = os.path.splitext(old_path)[1]
        
        new_name = new_name[:self.max_filename_length]
        print(f"Truncated new name: {new_name}")
        
        original_timestamp: str = self._extract_timestamp_from_filename(old_path)
        if original_timestamp is None:
            print(f"Warning: Could not extract timestamp from {old_path}")
            original_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        new_filename: str = f"{new_name}_{original_timestamp}{extension}"
        new_path: str = os.path.join(self.directory, new_filename)
        
        print(f"Attempting to rename: {old_path} -> {new_path}")
        try:
            os.rename(old_path, new_path)
            print(f"Successfully renamed: {old_path} -> {new_path}")
        except Exception as e:
            print(f"Error renaming {old_path}: {str(e)}")
            return  # Skip adding to group if renaming failed

        if self.group_files:
            self._add_to_group(new_path, original_timestamp)

    def _extract_timestamp_from_filename(self, filepath: str) -> str:
        """
        Extract timestamp from the filename.

        :param filepath: Path to the file.
        :return: Extracted timestamp as a string.
        """
        filename: str = os.path.basename(filepath)
        match = re.search(r'Screenshot (\d{4}-\d{2}-\d{2} at \d{2}\.\d{2}\.\d{2})', filename)
        if match:
            timestamp_str: str = match.group(1)
            timestamp: datetime = datetime.strptime(timestamp_str, '%Y-%m-%d at %H.%M.%S')
            return timestamp.strftime("%Y%m%d_%H%M%S")
        else:
            print(f"Could not extract timestamp from filename: {filename}")
            return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _add_to_group(self, file_path: str, timestamp: str) -> None:
        """
        Add a file to a group based on its timestamp.

        :param file_path: Path to the file.
        :param timestamp: Timestamp of the file.
        """
        file_time: datetime = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        group_key: Optional[str] = None

        for key in self.grouped_files.keys():
            # Extract timestamp from the key
            key_parts = key.split('_')
            if len(key_parts) >= 2:
                key_timestamp = '_'.join(key_parts[-2:])  # Join the last two parts
                try:
                    group_time: datetime = datetime.strptime(key_timestamp, "%Y%m%d_%H%M%S")
                    if abs(file_time - group_time) <= timedelta(minutes=self.group_time_threshold):
                        group_key = key
                        break
                except ValueError:
                    # If parsing fails, skip this key
                    continue

        if group_key is None:
            group_key = f"group_{timestamp}"

        if group_key not in self.grouped_files:
            self.grouped_files[group_key] = []

        # Add the new file to the group
        self.grouped_files[group_key].append(os.path.basename(file_path))

        # Sort the files in the group by their timestamps
        self.grouped_files[group_key] = self._sort_files_by_timestamp(self.grouped_files[group_key])

    def _sort_files_by_timestamp(self, files: List[str]) -> List[str]:
        """
        Sort files by their timestamps.

        :param files: List of filenames to sort.
        :return: Sorted list of filenames.
        """
        def extract_timestamp(filename: str) -> Tuple[datetime, str]:
            parts = filename.split('_')
            if len(parts) >= 2:
                try:
                    timestamp = datetime.strptime('_'.join(parts[-2:]), "%Y%m%d_%H%M%S")
                    return (timestamp, filename)
                except ValueError:
                    # If parsing fails, return a minimum datetime to sort it at the beginning
                    return (datetime.min, filename)
            return (datetime.min, filename)

        return [file for _, file in sorted(map(extract_timestamp, files), key=lambda x: x[0])]

    def write_grouped_files_to_json(self) -> None:
        """
        Write grouped files information to a JSON file in the specified directory.
        """
        if not self.group_files:
            print("File grouping is not enabled.")
            return

        timestamp = self.creation_time.strftime("%Y%m%d_%H%M%S")
        filename = f"grouped_screenshots_{timestamp}.json"
        output_file = os.path.join(self.directory, filename)

        with open(output_file, 'w') as f:
            json.dump(self.grouped_files, f, indent=2)
        print(f"Grouped files information written to {output_file}")

    def _merge_close_groups(self) -> None:
        """
        Merge groups that are within the group_time_threshold of each other.
        """
        merged_groups: Dict[str, List[str]] = {}
        sorted_keys = sorted(self.grouped_files.keys())

        for key in sorted_keys:
            key_time = self._extract_time_from_key(key)
            if key_time is None:
                continue

            merged = False
            for merged_key in merged_groups:
                merged_key_time = self._extract_time_from_key(merged_key)
                if merged_key_time is None:
                    continue

                if abs(key_time - merged_key_time) <= timedelta(minutes=self.group_time_threshold):
                    merged_groups[merged_key].extend(self.grouped_files[key])
                    merged_groups[merged_key] = self._sort_files_by_timestamp(merged_groups[merged_key])
                    merged = True
                    break

            if not merged:
                merged_groups[key] = self.grouped_files[key]

        self.grouped_files = merged_groups

    def _extract_time_from_key(self, key: str) -> Optional[datetime]:
        """
        Extract datetime from a group key.

        :param key: Group key string.
        :return: Extracted datetime or None if extraction fails.
        """
        parts = key.split('_')
        if len(parts) >= 2:
            try:
                return datetime.strptime('_'.join(parts[-2:]), "%Y%m%d_%H%M%S")
            except ValueError:
                return None
        return None

    def group_existing_files(self) -> None:
        """
        Group existing renamed files in the specified directory.
        """
        print(f"Grouping existing files in: {self.directory}")
        self.grouped_files = {}  # Reset grouped_files

        for filename in os.listdir(self.directory):
            if filename.endswith('.png'):
                filepath = os.path.join(self.directory, filename)
                timestamp = self._extract_timestamp_from_filename(filename)
                if timestamp:
                    self._add_to_group(filepath, timestamp)

        self._merge_close_groups()
        print("Grouping completed.")

    def _extract_timestamp_from_filename(self, filepath: str) -> Optional[str]:
        """
        Extract timestamp from the filename.

        :param filepath: Path to the file.
        :return: Extracted timestamp as a string, or None if not found.
        """
        filename: str = os.path.basename(filepath)
        # First, try to match the original Screenshot format
        match = re.search(r'Screenshot (\d{4}-\d{2}-\d{2} at \d{2}\.\d{2}\.\d{2})', filename)
        if match:
            timestamp_str: str = match.group(1)
            timestamp: datetime = datetime.strptime(timestamp_str, '%Y-%m-%d at %H.%M.%S')
            return timestamp.strftime("%Y%m%d_%H%M%S")
        else:
            # If not found, try to extract timestamp from the renamed format
            parts = filename.split('_')
            if len(parts) >= 2:
                try:
                    timestamp = '_'.join(parts[-2:]).split('.')[0]  # Remove file extension
                    datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                    return timestamp
                except ValueError:
                    pass
        print(f"Could not extract timestamp from filename: {filename}")
        return None

# Usage
if __name__ == "__main__":
    import sys

    directory = "/Users/alexiskirke/Dropbox/Screenwriting/MA Screenwriting - The Business of Film/One-Pager - Book Chapters"

    renamer = ScreenshotRenamer(group_files=True, group_time_threshold=15, directory=directory)

    renamer.rename_screenshots()
    renamer.group_existing_files()
    renamer.write_grouped_files_to_json()
    print("Process completed.")
