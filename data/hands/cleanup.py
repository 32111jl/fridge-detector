# deletes empty txt files and moves corresponding images to Testing
import os
import shutil

training_folder = 'Training'
testing_folder = 'Testing'

for txt_file in os.listdir(training_folder):
  if txt_file.endswith('.txt'):
    txt_path = os.path.join(training_folder, txt_file)
    
    if os.path.getsize(txt_path) == 0:
      image_file = txt_file.replace('.txt', '.jpg')
      image_path = os.path.join(training_folder, image_file)
      
      os.remove(txt_path)
      
      new_image_path = os.path.join(testing_folder, image_file)
      if os.path.exists(image_path) and not os.path.exists(new_image_path):
        shutil.move(image_path, new_image_path)

print("cleanup done")
