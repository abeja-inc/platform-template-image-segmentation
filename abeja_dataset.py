import re
from abeja.datasets import Client as DatasetsClient
from abeja.datalake import Client as DatalakeClient
import io
from PIL import Image
from torchvision.datasets.vision import VisionDataset

from utils import create_palette


def get_dataset_size(dataset_id):
        datasets_client = DatasetsClient()
        dataset = datasets_client.get_dataset(dataset_id)
        return dataset.total_count


class DataLakeObj:
    def __init__(self, channel_id: str, file_id: str, src_data):
        self.channel_id = channel_id
        self.file_id = file_id
        self.src_data = src_data


class AbejaDataset(VisionDataset):
    def __init__(self, 
                 root,
                 dataset_id,
                 transform=None,
                 target_transform=None,
                 transforms=None,
                 prefetch=False,
                 use_cache=True,
                 indices=None):
        
        super(AbejaDataset, self).__init__(root, transforms, transform, target_transform)

        datasets_client = DatasetsClient()
        self.datalake_client = DatalakeClient()
        dataset = datasets_client.get_dataset(dataset_id)
        self.labels = dataset.props['categories'][0]['labels']
        self.palette = create_palette(self.labels)
        self.use_cache = use_cache

        self.datalake_files = list()
        idx = 0
        for item in dataset.dataset_items.list(prefetch=prefetch):
            if indices is not None and not idx in indices:
                idx +=1
                continue

            if 'segmentation-image' in item.attributes:
                data_uri = item.attributes['segmentation-image']['combined']['data_uri']
            else:
                # FIXME: DEPRECATED. Type 'segmentation' is invalid on the latest spec.
                data_uri = item.attributes['segmentation']['combined']['data_uri']
            m = re.search(r'datalake://(.+?)/(.+?)$', data_uri)
            src_data = item.source_data[0]
            self.datalake_files.append(DataLakeObj(m.group(1), m.group(2), src_data))
            idx += 1

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        channel = self.datalake_client.get_channel(self.datalake_files[index].channel_id)
        datalake_file = channel.get_file(self.datalake_files[index].file_id)

        # source image
        src_data = self.datalake_files[index].src_data
        src_content = src_data.get_content(cache=self.use_cache)
        src_file_like_object = io.BytesIO(src_content)
        src_img = Image.open(src_file_like_object).convert('RGB')

        # target image
        content = datalake_file.get_content(cache=self.use_cache)
        file_like_object = io.BytesIO(content)
        target = Image.open(file_like_object).convert('RGB').quantize(palette=self.palette)

        if self.transforms is not None:
            src_img, target = self.transforms(src_img, target)

        return src_img, target

    def __len__(self):
        return len(self.datalake_files)

    def num_class(self):
        return len(self.labels)+1 # label+background
