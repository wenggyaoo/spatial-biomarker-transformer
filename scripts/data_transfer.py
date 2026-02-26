import os
import shutil
from pathlib import Path
from tqdm import tqdm

def transfer_s240_data(
    source_dir="/autofs/bal14/zqwu/CellularTables/CellularTables/s240",
    dest_dir="/autofs/bal14/khguo/data_celltype/train/s240",
    dry_run=False
):
    """
    Transfer s240 data with specific file renaming rules:
    1. *cell_data.5.csv -> cell_data.csv
    2. *expression.4.csv -> expression.csv
    3. *cell_types.8726-5-4.csv -> cell_type.csv
    4. Only transfer subfolders that contain *cell_types.8726-5-4.csv
    
    Args:
        source_dir: Source directory path
        dest_dir: Destination directory path
        dry_run: If True, only print actions without executing
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    if not source_path.exists():
        print(f"❌ Source directory does not exist: {source_dir}")
        return
    
    print(f"📁 Source: {source_dir}")
    print(f"📁 Destination: {dest_dir}")
    print(f"{'='*80}\n")
    
    # Find all subdirectories
    subdirs = [d for d in source_path.iterdir() if d.is_dir()]
    print(f"Found {len(subdirs)} subdirectories\n")
    
    # Analyze each subdirectory
    valid_subdirs = []
    skipped_subdirs = []
    
    for subdir in tqdm(subdirs, desc="Analyzing subdirectories"):
        # Check if subdirectory contains cell_types.8726-5-4.csv file
        cell_types_files = list(subdir.glob('*cell_types.8726-5-4.csv'))
        
        if not cell_types_files:
            skipped_subdirs.append(subdir.name)
            continue
        
        # Find all required files
        cell_data_files = list(subdir.glob('*cell_data.5.csv'))
        expression_files = list(subdir.glob('*expression.4.csv'))
        
        valid_subdirs.append({
            'subdir': subdir,
            'cell_types': cell_types_files[0] if cell_types_files else None,
            'cell_data': cell_data_files[0] if cell_data_files else None,
            'expression': expression_files[0] if expression_files else None
        })
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Valid subdirectories (contain cell_types file): {len(valid_subdirs)}")
    print(f"❌ Skipped subdirectories (no cell_types file): {len(skipped_subdirs)}")
    
    if skipped_subdirs:
        print(f"\n📋 Skipped subdirectories (first 20):")
        for name in skipped_subdirs[:20]:
            print(f"  - {name}")
        if len(skipped_subdirs) > 20:
            print(f"  ... and {len(skipped_subdirs) - 20} more")
    
    # Show what will be transferred
    print(f"\n{'='*80}")
    print(f"FILES TO TRANSFER")
    print(f"{'='*80}")
    
    for i, info in enumerate(valid_subdirs[:10], 1):
        print(f"\n{i}. Subdirectory: {info['subdir'].name}")
        if info['cell_types']:
            print(f"   ✓ {info['cell_types'].name} → cell_type.csv")
        if info['cell_data']:
            print(f"   ✓ {info['cell_data'].name} → cell_data.csv")
        if info['expression']:
            print(f"   ✓ {info['expression'].name} → expression.csv")
    
    if len(valid_subdirs) > 10:
        print(f"\n... and {len(valid_subdirs) - 10} more subdirectories")
    
    # Calculate total files
    total_files = 0
    for info in valid_subdirs:
        if info['cell_types']:
            total_files += 1
        if info['cell_data']:
            total_files += 1
        if info['expression']:
            total_files += 1
    
    print(f"\n📦 Total files to transfer: {total_files}")
    
    if dry_run:
        print(f"\n🔍 DRY RUN MODE - No files were actually transferred")
        print(f"Set dry_run=False to perform the actual transfer")
        return valid_subdirs, skipped_subdirs
    
    # Ask for confirmation
    print(f"\n{'='*80}")
    response = input(f"Proceed with transfer? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("❌ Transfer cancelled")
        return valid_subdirs, skipped_subdirs
    
    # Perform transfer
    print(f"\n🚀 Starting transfer...")
    success_count = 0
    error_count = 0
    errors = []
    
    for info in tqdm(valid_subdirs, desc="Transferring subdirectories"):
        subdir_name = info['subdir'].name
        dest_subdir = dest_path / subdir_name
        
        try:
            # Create destination subdirectory
            dest_subdir.mkdir(parents=True, exist_ok=True)
            
            # Transfer and rename cell_types file
            if info['cell_types']:
                dest_file = dest_subdir / 'cell_type.csv'
                shutil.copy2(info['cell_types'], dest_file)
                success_count += 1
            
            # Transfer and rename cell_data file
            if info['cell_data']:
                dest_file = dest_subdir / 'cell_data.csv'
                shutil.copy2(info['cell_data'], dest_file)
                success_count += 1
            
            # Transfer and rename expression file
            if info['expression']:
                dest_file = dest_subdir / 'expression.csv'
                shutil.copy2(info['expression'], dest_file)
                success_count += 1
                
        except Exception as e:
            error_msg = f"Error processing {subdir_name}: {e}"
            errors.append(error_msg)
            error_count += 1
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"TRANSFER COMPLETE")
    print(f"{'='*80}")
    print(f"✅ Successfully transferred: {success_count} files")
    print(f"📁 Subdirectories processed: {len(valid_subdirs)}")
    
    if error_count > 0:
        print(f"\n❌ Errors encountered: {error_count}")
        for error in errors[:10]:
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    print(f"{'='*80}")
    
    return valid_subdirs, skipped_subdirs


# ========== USAGE ==========

if __name__ == "__main__":
    # Step 1: Dry run to see what will happen
    print("Step 1: Running in DRY RUN mode...\n")
    valid, skipped = transfer_s240_data(dry_run=False)
    
    # Step 2: Uncomment to actually perform the transfer
    # print("\n\nStep 2: Performing actual transfer...\n")
    # valid, skipped = transfer_s240_data(dry_run=False)