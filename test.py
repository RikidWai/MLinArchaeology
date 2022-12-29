import os
rootdir = '/content/drive/MyDrive/export_images'
targetdir = '/content/drive/MyDrive/CS_FYP_archaeology/test_images/'
# Loop over differ
for root, dirs, files in os.walk(rootdir):
    for file in files:
        filename, extension = os.path.splitext(file)
        if 'cr' in extension or 'CR' in extension:
            path = os.path.join(root, file)
            dir = root.split(os.path.sep)[-1]
            print(f'{dir}_{filename}_s{1}')
