import os
import pickle
import shutil

out_dir = "./output_dir" 
key_dir = "./keys"          
os.makedirs(key_dir, exist_ok=True)

# Step 0: Remove unwanted files and folders
for item in os.listdir(out_dir):
    # Remove files starting with "._"
    if item.startswith("._"):
        file_path = os.path.join(out_dir, item)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Removed file {file_path}")
    # Remove any __MACOSX folder entirely
    if item == "__MACOSX":
        dir_path = os.path.join(out_dir, item)
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
            print(f"Removed directory {dir_path}")

# Step 1: Rename files to add .pkl suffix if not already present
for file_name in os.listdir(out_dir):
    file_path = os.path.join(out_dir, file_name)
    if os.path.isfile(file_path) and not file_name.endswith(".pkl"):
        new_file_path = file_path + ".pkl"
        os.rename(file_path, new_file_path)
        print(f"Renamed {file_path} -> {new_file_path}")

# Step 2: Collect all .pkl files
all_files = sorted([f for f in os.listdir(out_dir) if f.endswith(".pkl")])
print("Total .pkl files:", len(all_files))

# Step 3: Split into train, validation, and test datasets (80/10/10 split)
num_total = len(all_files)
num_train = int(num_total * 0.80)
num_val = int(num_total * 0.10)

train_files = all_files[:num_train]
valid_files = all_files[num_train:num_train + num_val]
test_files = all_files[num_train + num_val:]

print(f"Train files: {len(train_files)}, Valid files: {len(valid_files)}, Test files: {len(test_files)}")

# Step 4: Save the keys into pickle files in the key_dir
with open(os.path.join(key_dir, "train_keys.pkl"), "wb") as f:
    pickle.dump(train_files, f)
with open(os.path.join(key_dir, "valid_keys.pkl"), "wb") as f:
    pickle.dump(valid_files, f)
with open(os.path.join(key_dir, "test_keys.pkl"), "wb") as f:
    pickle.dump(test_files, f)

print("Keys saved in:", key_dir)
