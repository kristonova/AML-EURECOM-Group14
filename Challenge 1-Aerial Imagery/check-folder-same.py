import os
import hashlib

def md5sum(filename, block_size=65536):
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            md5.update(chunk)
    return md5.hexdigest()

def get_files_with_hashes(directory):
    """Return a dictionary {filename: md5} for files in a directory."""
    files_hashes = {}
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            files_hashes[file] = md5sum(path)
    return files_hashes

def compare_directories(dir1, dir2):
    """Compare files (names and content) between two directories."""
    files1 = get_files_with_hashes(dir1)
    files2 = get_files_with_hashes(dir2)
    
    names1 = set(files1.keys())
    names2 = set(files2.keys())
    
    if names1 != names2:
        print("File names differ between directories.")
        diff1 = names1 - names2
        diff2 = names2 - names1
        if diff1:
            print(f"Files in {dir1} but not in {dir2}: {diff1}")
        if diff2:
            print(f"Files in {dir2} but not in {dir1}: {diff2}")
        return False
    else:
        all_same = True
        for filename in names1:
            if files1[filename] != files2[filename]:
                print(f"File {filename} differs.")
                all_same = False
        if all_same:
            print("All files are identical between directories.")
        return all_same

if __name__ == "__main__":
    dir1 = "archive/test/test"
    dir2 = "archive/train/train"
    compare_directories(dir1, dir2)