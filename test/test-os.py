import os

path, file_name = os.path.split('images/out.png')
print(path)
print(file_name)

new_path = os.path.join(path, 'Y_U_V', 'Y' + file_name)
print(new_path)