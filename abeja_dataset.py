import re
from abeja.datasets import Client as DatasetsClient
from abeja.datalake import Client as DatalakeClient
import os
import sys
import collections
import io
from PIL import Image
from torchvision.datasets.vision import VisionDataset


def getDatasetSize(dataset_id):
        datasets_client = DatasetsClient()
        dataset = datasets_client.get_dataset(dataset_id)
        return dataset.total_count
    
class AbejaDataset(VisionDataset):
    def __init__(self, 
                root,
                dataset_id,
                transform=None,
                target_transform=None,
                transforms=None,
                indices=None):
        
        super(AbejaDataset, self).__init__(root, transforms, transform, target_transform)

        datasets_client = DatasetsClient()
        self.datalake_client = DatalakeClient()
        dataset = datasets_client.get_dataset(dataset_id)
        self.labels = dataset.props['categories'][0]['labels']

        self.datalake_files = list()
        idx = 0
        for item in dataset.dataset_items.list(prefetch=False):
            if indices is not None and not idx in indices:
                idx +=1
                continue
            # 'combined.data_uri' は (多分) category内の全てのlabelをひとつに纏めた画像
            # 'layers[].data_uri' は特定の 'label_id' のみの画像
            data_uri = item.attributes['segmentation-image']['combined']['data_uri']
            m = re.search(r'datalake://(.+?)/(.+?)$', data_uri)
            src_data = item.source_data[0]
            self.datalake_files.append(((m.group(1),m.group(2)), src_data))
            idx += 1


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        channel = self.datalake_client.get_channel(self.datalake_files[index][0][0])
        datalake_file = channel.get_file(self.datalake_files[index][0][1])

        # source image
        src_data = self.datalake_files[index][1]
        src_content = src_data.get_content(cache=True)
        src_file_like_object = io.BytesIO(src_content)
        src_img = Image.open(src_file_like_object).convert('RGB')

        # target image
        content = datalake_file.get_content(cache=True)
        file_like_object = io.BytesIO(content)
        target = Image.open(file_like_object).convert('P')

        if self.transforms is not None:
            src_img, target = self.transforms(src_img, target)

        return src_img, target


    def __len__(self):
        return len(self.datalake_files)

    def num_class(self):
        return len(self.labels)+1 # label+background
    


