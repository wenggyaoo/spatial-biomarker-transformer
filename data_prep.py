import os
import shutil
import re

# --- USER CONFIGURATION ---
# 1. Specify the single, top-level folder that contains all your study subfolders.
# The script will look for folders like 'Experiment_A', 'Experiment_B', etc., inside this directory.
# Example for Windows: 'C:/Users/User/Desktop/AllMyExperiments'
# Example for macOS/Linux: '/home/user/data/AllStudies'
ROOT_SOURCE_DIRECTORY = r'C:\Users\alexk\PycharmProjects\Model_v1\.venv\raw_data'

# 2. Specify the main folder where you want the final, organized output to go.
# The script will create subfolders here matching the names of your source study folders.
# Example for Windows: 'C:/Users/User/Desktop/SortedOutput'
# Example for macOS/Linux: '/home/user/data/ProcessedOutput'
DESTINATION_DIRECTORY = r'C:\Users\alexk\PycharmProjects\Model_v1\.venv\data\train'


# --------------------------


def process_and_sort_study_files(source_study_dir, dest_study_dir):
    """
    Scans a study folder, moves files into sorted subfolders, and renames them correctly.
    - Groups files by the full 'c..._reg...' code.
    - Renames files to a standard format like 'cell_data.csv'.

    Args:
        source_study_dir (str): The full path to the source study folder.
        dest_study_dir (str): The full path to the destination study folder.
    """
    print(f"   -> Reading files from: {source_study_dir}")

    os.makedirs(dest_study_dir, exist_ok=True)

    for filename in os.listdir(source_study_dir):
        source_path = os.path.join(source_study_dir, filename)

        if os.path.isfile(source_path):
            # Rule 1: Discard 'cell_features' files.
            if 'cell_features' in filename:
                print(f"      - Discarding: {filename}")
                continue

            # Rule 2: Find the full experiment code to determine the destination folder.
            match = re.search(r'(c\d{3}_v\d{3}_r\d{3}_reg\d{3})', filename)

            if match:
                grouping_code = match.group(1)
                final_grouping_dir = os.path.join(dest_study_dir, grouping_code)
                os.makedirs(final_grouping_dir, exist_ok=True)

                # --- CORRECTED RENAMING LOGIC ---
                # First, get the filename without its final extension (e.g., '.csv').
                base_name = os.path.splitext(filename)[0]

                # Now, from the base_name, split by '.' and get the last part, which is the data type.
                try:
                    # e.g., '...reg001.cell_data' becomes 'cell_data'
                    data_type_part = base_name.split('.')[-1]
                except IndexError:
                    print(f"      - Skipping '{filename}': Cannot determine data type from filename format.")
                    continue

                # Standardize 'cell_types' to 'cell_type'.
                if data_type_part == 'cell_types':
                    new_name_base = 'cell_type'
                else:
                    new_name_base = data_type_part

                # Construct the final, clean filename.
                new_filename = f"{new_name_base}.csv"
                # --- END OF CORRECTED LOGIC ---

                # Define the final path for the file, including its new name.
                destination_path = os.path.join(final_grouping_dir, new_filename)

                # Move the file to the new location and apply the new name.
                print(f"      - Moving '{filename}' to '{destination_path}'")
                shutil.move(source_path, destination_path)
            else:
                print(f"      - Skipping '{filename}': No matching code found or discard rule match.")


# --- Main execution block ---
if __name__ == "__main__":
    if 'path/to/your/' in ROOT_SOURCE_DIRECTORY or 'path/to/your/' in DESTINATION_DIRECTORY:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! PLEASE UPDATE THE 'ROOT_SOURCE_DIRECTORY' AND      !!!")
        print("!!! 'DESTINATION_DIRECTORY' VARIABLES BEFORE RUNNING.  !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        print("--- Starting Hierarchical File Organization and Renaming ---")

        os.makedirs(DESTINATION_DIRECTORY, exist_ok=True)
        print(f"Output will be saved in: {DESTINATION_DIRECTORY}\n")

        for study_name in os.listdir(ROOT_SOURCE_DIRECTORY):
            source_study_path = os.path.join(ROOT_SOURCE_DIRECTORY, study_name)

            if os.path.isdir(source_study_path):
                print(f"--- Processing Study: {study_name} ---")

                dest_study_path = os.path.join(DESTINATION_DIRECTORY, study_name)

                process_and_sort_study_files(source_study_path, dest_study_path)

                print(f"--- Finished Study: {study_name} ---\n")
            else:
                print(f"--- Skipping non-directory item: {study_name} ---\n")

        print("==============================================")
        print("All studies have been processed successfully.")
        print("==============================================")