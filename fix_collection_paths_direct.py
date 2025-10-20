#!/usr/bin/env python3
"""
Fix ChromaDB collection metadata paths by directly updating SQLite database.

This script updates the json_source path in collection metadata
when the project directory has been renamed.
"""
import sys
import sqlite3
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from common.settings import get_settings


def fix_collection_paths_direct(old_dir_name: str, new_dir_name: str, dry_run: bool = True):
    """
    Update collection metadata paths by directly modifying SQLite database.
    
    Args:
        old_dir_name: Old directory name (e.g., 'PageIndex')
        new_dir_name: New directory name (e.g., 'ExRAG')
        dry_run: If True, only show what would be changed without making changes
    """
    settings = get_settings()
    chroma_db_path = settings.chroma_db_dir / "chroma.sqlite3"
    
    if not chroma_db_path.exists():
        print(f"❌ ChromaDB database not found at: {chroma_db_path}")
        return
    
    print(f"{'🔍 DRY RUN - ' if dry_run else ''}Connecting to: {chroma_db_path}\n")
    print("="*80)
    
    # Connect to SQLite database
    conn = sqlite3.connect(str(chroma_db_path))
    cursor = conn.cursor()
    
    # Get all collections
    cursor.execute("SELECT id, name FROM collections")
    collections = cursor.fetchall()
    
    if not collections:
        print("❌ No collections found in database.")
        conn.close()
        return
    
    print(f"Found {len(collections)} collection(s)\n")
    
    fixed_count = 0
    skipped_count = 0
    
    for collection_id, collection_name in collections:
        print(f"\n📦 Collection: {collection_name}")
        
        # Get metadata for this collection
        cursor.execute(
            "SELECT key, str_value FROM collection_metadata WHERE collection_id = ? AND key = 'json_source'",
            (collection_id,)
        )
        result = cursor.fetchone()
        
        if result:
            key, old_path = result
            
            if old_dir_name in old_path:
                new_path = old_path.replace(old_dir_name, new_dir_name)
                
                print(f"   ❌ Old path: {old_path}")
                print(f"   ✅ New path: {new_path}")
                
                # Verify new path exists
                if Path(new_path).exists():
                    print(f"   ✓ New path verified: File exists")
                    
                    if not dry_run:
                        # Update metadata
                        cursor.execute(
                            "UPDATE collection_metadata SET str_value = ? WHERE collection_id = ? AND key = 'json_source'",
                            (new_path, collection_id)
                        )
                        print(f"   ✓ Updated successfully!")
                        fixed_count += 1
                    else:
                        print(f"   🔍 Would update (dry run)")
                        fixed_count += 1
                else:
                    print(f"   ⚠️  Warning: New path does not exist!")
                    print(f"   💡 Skipping this collection")
                    skipped_count += 1
            else:
                print(f"   ✓ Path is correct (no old directory name found)")
                print(f"   Current path: {old_path}")
                skipped_count += 1
        else:
            print(f"   ⚠️  No json_source in metadata")
            skipped_count += 1
    
    # Commit changes if not dry run
    if not dry_run:
        conn.commit()
        print(f"\n✅ Database changes committed")
    
    conn.close()
    
    print("\n" + "="*80)
    print(f"\n📊 Summary:")
    print(f"   {'Would fix' if dry_run else 'Fixed'}: {fixed_count} collection(s)")
    print(f"   Skipped: {skipped_count} collection(s)")
    
    if dry_run and fixed_count > 0:
        print(f"\n💡 Run with --apply to actually update the collections")
    elif not dry_run and fixed_count > 0:
        print(f"\n✅ All collections updated successfully!")
    print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fix ChromaDB collection paths by directly updating SQLite database"
    )
    parser.add_argument(
        "--old",
        required=True,
        help="Old directory name (e.g., 'PageIndex')"
    )
    parser.add_argument(
        "--new",
        required=True,
        help="New directory name (e.g., 'ExRAG')"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply the changes (default is dry run)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🔧 ChromaDB Collection Path Fixer (Direct SQLite)")
    print("="*80)
    print(f"Old directory name: {args.old}")
    print(f"New directory name: {args.new}")
    print(f"Mode: {'APPLY CHANGES' if args.apply else 'DRY RUN (no changes)'}")
    print("="*80 + "\n")
    
    if not args.apply:
        print("⚠️  Running in DRY RUN mode - no changes will be made")
        print("   Use --apply to actually update the collections\n")
    else:
        print("⚠️  This will directly modify the ChromaDB SQLite database")
        print("   Make sure to backup your database if needed\n")
    
    try:
        fix_collection_paths_direct(args.old, args.new, dry_run=not args.apply)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

