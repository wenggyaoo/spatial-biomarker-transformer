import os
import shutil


def process_dataset(raw_data_dir, processed_data_dir):
    """
    Processes a dataset of CSV files organized in study folders.

    This function reads from a source directory, filters files based on
    naming conventions, and reorganizes them into a new directory structure.

    The expected input file naming convention is:
    'studyName_regionID_contentIdentifier.csv'
    Example: 'Charville_c001_v001_r001_reg001_expression.csv'

    The output structure will be:
    processed_data_dir/studyName/regionID/contentIdentifier.csv
    Example: 'processed_data/Charville/c001_v001_r001_reg001/expression.csv'

    Args:
        raw_data_dir (str): The path to the main folder containing the raw dataset.
        processed_data_dir (str): The path to the folder where the processed
                                  data will be saved.
    """
    # --- 1. Setup: Create the main processed data directory ---
    print(f"Starting dataset processing.")
    print(f"Raw data source: '{raw_data_dir}'")
    print(f"Processed data destination: '{processed_data_dir}'")

    if not os.path.isdir(raw_data_dir):
        print(f"Error: Raw data directory not found at '{raw_data_dir}'")
        return

    try:
        os.makedirs(processed_data_dir, exist_ok=True)
        print(f"Successfully created or found destination folder: '{processed_data_dir}'")
    except OSError as e:
        print(f"Error creating destination directory {processed_data_dir}: {e}")
        return

    # Content identifiers to keep
    valid_content_identifiers = {"cell_data", "expression"}

    # --- 2. Iterate through each study folder in the raw data directory ---
    for study_name in os.listdir(raw_data_dir):
        source_study_path = os.path.join(raw_data_dir, study_name)

        if os.path.isdir(source_study_path):
            print(f"\nProcessing study: '{study_name}'")

            # Create corresponding study folder in the processed data directory
            dest_study_path = os.path.join(processed_data_dir, study_name)
            os.makedirs(dest_study_path, exist_ok=True)

            # --- 3. Iterate through each file in the study folder ---
            for filename in os.listdir(source_study_path):
                source_file_path = os.path.join(source_study_path, filename)

                # Process only .csv files
                if filename.endswith('.csv'):
                    try:
                        # --- 4. Parse the filename to extract components ---
                        # Remove the '.csv' extension
                        base_name = os.path.splitext(filename)[0]

                        # Split from the right to get the content identifier
                        # e.g., 'Charville_c001_v001_r001_reg001_expression' ->
                        # ['Charville_c001_v001_r001_reg001', 'expression']
                        parts = base_name.rsplit('_', 1)
                        if len(parts) != 2:
                            print(f"  - Skipping file with unexpected name format: {filename}")
                            continue

                        prefix, content_identifier = parts

                        # --- 5. Filter based on content identifier ---
                        if content_identifier in valid_content_identifiers:
                            # Split the prefix to separate the study name from the region ID
                            # e.g., 'Charville_c001_v001_r001_reg001' ->
                            # ['Charville', 'c001_v001_r001_reg001']
                            region_parts = prefix.split('_', 1)
                            if len(region_parts) != 2:
                                print(f"  - Skipping file with unexpected prefix format: {filename}")
                                continue

                            # The region folder name is the second part
                            region_folder_name = region_parts[1]

                            # --- 6. Create new structure and copy the file ---
                            # Define the new region folder path
                            dest_region_path = os.path.join(dest_study_path, region_folder_name)
                            os.makedirs(dest_region_path, exist_ok=True)

                            # Define the final destination path for the renamed file
                            new_filename = f"{content_identifier}.csv"
                            dest_file_path = os.path.join(dest_region_path, new_filename)

                            # Copy the file to the new location with the new name
                            shutil.copy(source_file_path, dest_file_path)
                            print(
                                f"  + Processed and copied: {filename} -> {os.path.join(region_folder_name, new_filename)}")

                    except Exception as e:
                        print(f"  - An error occurred while processing {filename}: {e}")

    print("\nDataset processing complete.")


if __name__ == '__main__':
    # --- HOW TO USE ---
    # 1. Set the path to your main data folder below.
    #    This folder should contain the study folders.
    #    Example for Windows: 'C:\\Users\\YourUser\\Desktop\\my_dataset'
    #    Example for macOS/Linux: '/home/user/my_dataset'
    raw_dataset_directory = 'raw_data'

    # 2. Set the path where you want the processed data to be saved.
    #    This folder will be created if it doesn't exist.
    processed_dataset_directory = 'processed_data'


    # 3. Create a dummy folder structure for testing if you don't have the data yet.
    #    You can uncomment the function below to create a sample dataset.
    def create_dummy_raw_data(base_dir):
        print(f"Creating a dummy raw dataset at '{base_dir}' for demonstration.")
        os.makedirs(base_dir, exist_ok=True)
        studies = {'StudyA': 'Charville', 'StudyB': 'Bordeaux'}
        for study_key, study_prefix in studies.items():
            study_path = os.path.join(base_dir, study_key)
            os.makedirs(study_path, exist_ok=True)
            for i in range(1, 3):  # Create 2 regions per study
                region_id = f"c00{i}_v001_r001_reg00{i}"
                # Create the valid files
                with open(os.path.join(study_path, f"{study_prefix}_{region_id}_expression.csv"), 'w') as f:
                    f.write("gene,value\nGENE1,100")
                with open(os.path.join(study_path, f"{study_prefix}_{region_id}_cell_data.csv"), 'w') as f:
                    f.write("cell,type\nCELL1,T-cell")
                # Create a file that should be ignored
                with open(os.path.join(study_path, f"{study_prefix}_{region_id}_metadata.json"), 'w') as f:
                    f.write("{}")
                # Create a zip file that should be ignored
                with open(os.path.join(study_path, f"{study_prefix}_{region_id}_images.zip"), 'w') as f:
                    f.write("zip_content")
        print("Dummy dataset created.")


    # Uncomment the line below to create a sample 'raw_data' folder for testing
    # create_dummy_raw_data(raw_dataset_directory)

    # Run the main processing function
    process_dataset(raw_dataset_directory, processed_dataset_directory)