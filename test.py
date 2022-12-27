import os
rootdir = 'C:/Users/sid/Desktop/test'

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print(os.path.join(subdir, file))