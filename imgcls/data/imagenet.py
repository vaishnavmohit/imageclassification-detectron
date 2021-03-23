import os
import os.path as osp
import json
from detectron2.data import DatasetCatalog, MetadataCatalog


def register_imagenet_instances(name, metadata, json_file):

    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file

    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: json.load(open(json_file)))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, evaluator_type="imagenet", **metadata
    )


# Update info
_root = os.getenv("DETECTRON2_DATASETS", "/users/mvaishn1/data/data/mvaishn1/fossil/ImageClassification/datasets/")
imagenet_train_annotation_file = osp.join(_root, "ImageNet2012/imagenet_detectron2_train.json")
imagenet_val_annotation_file = osp.join(_root, "ImageNet2012/imagenet_detectron2_val.json")

# Register into DatasetCatalog
register_imagenet_instances("imagenet_train", {}, imagenet_train_annotation_file)
register_imagenet_instances("imagenet_val", {}, imagenet_val_annotation_file)


