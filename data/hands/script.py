import os
import shutil

source_dir = "hands/Hands"
training_dir = "hands/Training"
testing_dir = "hands/Testing"

for file_name in os.listdir(source_dir):
  if file_name.endswith(".jpg"):
    base_name = os.path.splitext(file_name)[0]
    
    txt_file_path = os.path.join(source_dir, base_name + ".txt")
    img_file_path = os.path.join(source_dir, file_name)
    
    # if txt file exists, move to training
    if os.path.exists(txt_file_path):
      shutil.move(img_file_path, os.path.join(training_dir, file_name))
      shutil.move(txt_file_path, os.path.join(training_dir, base_name + ".txt"))
    
    # move to testing
    else:
      shutil.move(img_file_path, os.path.join(testing_dir, file_name))


print("files sorted into Training/Testing folders")
