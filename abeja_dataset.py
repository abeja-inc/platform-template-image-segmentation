mport re
from abeja.datasets import Client as DatasetsClient
from abeja.datalake import Client as DatalakeClient
import os
import sys
import collections
import io
from PIL import Image
from torchvision.datasets.vision import VisionDataset

class AbejaDataset(VisionDataset):
    def __init__(self, 
                root,
                user_id,
                personal_access_token,
                organization_id,
                dataset_id,
                early_stopping_test_size = 0,
                train_data = True,
                transform=None,
                target_transform=None,
                transforms=None):
        
        super(AbejaDataset, self).__init__(root, transforms, transform, target_transform)

        # set credential
        credential = {
            'user_id': user_id,
            'personal_access_token': personal_access_token
        }

        datasets_client = DatasetsClient(organization_id, credential)
        self.datalake_client = DatalakeClient(organization_id, credential)
        dataset = datasets_client.get_dataset(dataset_id)
        self.labels = dataset.props['categories'][0]['labels']

        datalake_files = list()
        for item in dataset.dataset_items.list(prefetch=False):
            # 'combined.data_uri' は (多分) category内の全てのlabelをひとつに纏めた画像
            # 'layers[].data_uri' は特定の 'label_id' のみの画像
            data_uri = item.attributes['segmentation-image']['combined']['data_uri']
            m = re.search(r'datalake://(.+?)/(.+?)$', data_uri)
            src_data = item.source_data[0]
            datalake_files.append(((m.group(1),m.group(2)), src_data))

        self.datalake_files = list()
        test_size = int(len(datalake_files) * early_stopping_test_size)
        if(test_size):
            self.datalake_files = datalake_files[test_size:] if train_data else datalake_files[:test_size]
        else:
            if(train_data is False):
                raise Exception("Dataset size is too small. Please add more dataset.")
            self.datalake_files = datalake_files



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
        src_img = Image.open(src_file_like_object)

        # target image
        content = datalake_file.get_content(cache=True)
        file_like_object = io.BytesIO(content)
        target = Image.open(file_like_object)

        if self.transforms is not None:
            src_img, target = self.transforms(src_img, target)

        return src_img, target


    def __len__(self):
        return len(self.datalake_files)

    def num_class(self):
        return len(self.labels)
    


