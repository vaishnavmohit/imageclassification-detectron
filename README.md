<!--
 * @Copyright (c) tkianai All Rights Reserved.
 * @Author         : vaishnavmohit
 * @Github         : https://github.com/vaishnavmohit
 -->

# changes made by MOHIT
Change the file at the below mentioned location to add normalization of image:

/env/detectron2_fossil/lib/python3.7/site-packages/detectron2/data/detection_utils.py

# ImageClassification.detectron2

Image classification based on detectron2.

This provides a convenient way to initialize backbone in detectron2.


## Usage

- Trained with detectron2 builtin trainer

1. Use default data flow in detectron2, you only need rename `forward_d2` to `forward`, while renaming `forward` to `forward_imgnet` in `imgcls/modeling/meta_arch/clsnet.py`

2. Create your own model config

3. Run: `python train_net_builtin.py --num-gpus <gpu number> --config-file configs/<your config file>`. For example: `sh scripts/train_net_builtin.sh`


- Trained with pytorch formal imagenet trainer [**Recommend**]

1. Read carefully with some arguments in `train_net.py`
2. Run: `sh /scripts/train_net.sh`

### Originally adapted from:

https://github.com/tkianai/ImageClassification.detectron2