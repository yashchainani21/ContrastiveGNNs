"""
Cleanup script to remove all .pgnet DORAnet network files from the repository.

These files are generated during DORAnet chemoenzymatic modification of polyketide
structures and can accumulate over time, taking up significant disk space.

Usage:
    python scripts/cleanup_pgnet_files.py           # Interactive mode
    python scripts/cleanup_pgnet_files.py --dry-run # Preview what would be deleted
    python scripts/cleanup_pgnet_files.py -y        # Delete without confirmation
"""

import argparse
from pathlib import Path
import sys


def find_pgnet_files(search_dir: Path, recursive: bool = True) -> list[Path]:
    """
    Find all .pgnet files in the specified directory.

    Args:
        search_dir: Directory to search in.
        recursive: If True, search recursively in subdirectories.

    Returns:
        List of Path objects for .pgnet files.
    """
    if recursive:
        pgnet_files = list(search_dir.rglob("*.pgnet"))
    else:
        pgnet_files = list(search_dir.glob("*.pgnet"))

    return sorted(pgnet_files)


def format_file_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def main():
    # Automatically detect repository root (parent of scripts directory)
    REPO_ROOT = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Remove all .pgnet DORAnet network files from the ContrastiveGNNs repository"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help=f"Directory to search for .pgnet files (default: repository root at {REPO_ROOT})"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Search recursively in subdirectories (default: True)"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only search in the specified directory, not subdirectories"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt and delete immediately"
    )

    args = parser.parse_args()

    # Handle recursive flag
    recursive = args.recursive and not args.no_recursive

    # Resolve search directory - default to repository root
    if args.dir is None:
        search_dir = REPO_ROOT
    else:
        search_dir = Path(args.dir).resolve()

    if not search_dir.exists():
        print(f"Error: Directory '{search_dir}' does not exist.")
        sys.exit(1)

    if not search_dir.is_dir():
        print(f"Error: '{search_dir}' is not a directory.")
        sys.exit(1)

    print(f"Searching for .pgnet files in: {search_dir}")
    print(f"Recursive search: {recursive}")
    print()

    # Find all .pgnet files
    pgnet_files = find_pgnet_files(search_dir, recursive=recursive)

    if not pgnet_files:
        print("No .pgnet files found. Nothing to clean up!")
        return

    # Calculate total size
    total_size = sum(f.stat().st_size for f in pgnet_files)

    # Display summary
    print(f"Found {len(pgnet_files)} .pgnet file(s)")
    print(f"Total size: {format_file_size(total_size)}")
    print()

    # List files by category
    enzymatic_files = [f for f in pgnet_files if 'enzymatic' in f.name.lower()]
    synthetic_files = [f for f in pgnet_files if 'synthetic' in f.name.lower()]
    other_files = [f for f in pgnet_files if 'enzymatic' not in f.name.lower() and 'synthetic' not in f.name.lower()]

    if enzymatic_files:
        enzymatic_size = sum(f.stat().st_size for f in enzymatic_files)
        print(f"  Enzymatic networks: {len(enzymatic_files)} files ({format_file_size(enzymatic_size)})")

    if synthetic_files:
        synthetic_size = sum(f.stat().st_size for f in synthetic_files)
        print(f"  Synthetic networks: {len(synthetic_files)} files ({format_file_size(synthetic_size)})")

    if other_files:
        other_size = sum(f.stat().st_size for f in other_files)
        print(f"  Other networks: {len(other_files)} files ({format_file_size(other_size)})")

    print()

    # Show sample files (first 10 and last 10 if more than 20)
    if len(pgnet_files) <= 20:
        print("Files to be deleted:")
        for f in pgnet_files:
            print(f"  - {f.relative_to(search_dir)} ({format_file_size(f.stat().st_size)})")
    else:
        print("Sample files to be deleted (showing first 10 and last 10):")
        for f in pgnet_files[:10]:
            print(f"  - {f.relative_to(search_dir)} ({format_file_size(f.stat().st_size)})")
        print(f"  ... ({len(pgnet_files) - 20} more files) ...")
        for f in pgnet_files[-10:]:
            print(f"  - {f.relative_to(search_dir)} ({format_file_size(f.stat().st_size)})")

    print()

    # Dry run mode
    if args.dry_run:
        print("DRY RUN MODE: No files will be deleted.")
        print(f"Would delete {len(pgnet_files)} files ({format_file_size(total_size)})")
        return

    # Confirmation prompt (unless --yes flag is used)
    if not args.yes:
        response = input(f"Delete {len(pgnet_files)} .pgnet files? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("Cancelled. No files were deleted.")
            return

    # Delete files
    print(f"\nDeleting {len(pgnet_files)} files...")
    deleted_count = 0
    failed_count = 0

    for f in pgnet_files:
        try:
            f.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"  Failed to delete {f.name}: {e}")
            failed_count += 1

    # Summary
    print()
    print("=" * 60)
    print("Cleanup complete!")
    print(f"  Deleted: {deleted_count} files ({format_file_size(total_size)})")
    if failed_count > 0:
        print(f"  Failed: {failed_count} files")
    print("=" * 60)


if __name__ == "__main__":
    main()
