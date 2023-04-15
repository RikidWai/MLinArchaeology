# Functions currently not in use but may be useful for Pytorch data handling


# Custom dataset inheriting the Pytorch generic Dataset
# Use this for higher flexibility, otherwise use ImageFolder for convenience
# Can modify the __getitem__ to customize the data structure returned from each sample
# Works for single folder containing data of all classes, uses csv_file to retrieve label for each image
class SherdDataSet(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
      """
      Args:
          csv_file (string): Path to the csv file with (img_path, label) for each row.
          root_dir (string): Directory with all the images.
          transform (callable, optional): Optional transform to be applied
              on a sample.
      """
      self.sherds_frame = pd.read_csv(csv_file)
      self.root_dir = root_dir
      self.transform = transform

    def __len__(self):
      return len(self.sherds_frame)

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()

      img_name = os.path.join(self.root_dir, self.sherds_frame.iloc[idx, 0])
      sherd_img = cv2.imread(img_name)
      sherd_label = self.sherds_frame.iloc[idx, 1]

      if self.transform:
          sample = self.transform(sherd_img)

      sample = {'image': sherd_img, 'landmarks': sherd_label}

      return sample


# Distributes data into subfolders by class
# Assumes all data in one single root folder initially
# root/label1/xxx.jpg
# root/label1/xxy.jpg
# root/label2/mmm.jpg
# root/label2/mmn.jpg
def distribute_by_class(datadir, csv_path):
  root = datadir
  my_csv_file = csv_path

  # Loading csv as {image:class,...} format
  df = pd.read_csv(my_csv_file).set_index('images')
  class_dict = df.idxmax(axis="columns").to_dict()

  for path in Path(root).iterdir():
    if path.is_file() and path.name in class_dict.keys():
      path.rename(Path(path.parent, class_dict[path.name], path.name))

