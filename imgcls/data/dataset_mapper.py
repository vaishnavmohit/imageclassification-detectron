from . import classification_utils as c_utils


from detectron2.data.dataset_mapper import DatasetMapper as _DatasetMapper

import detectron2.data.transforms as T

class DatasetMapper(_DatasetMapper):

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        self.tfm_gens = c_utils.build_transform_gen(cfg, is_train)