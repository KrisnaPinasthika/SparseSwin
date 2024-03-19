# SparseSwin as an Object Detector

SparseSwin is an architecture designed to tackle image classification tasks.
However, its development is not limited to solving image classification alone.
In this research, we extended the architecture to function as an object detector.

Note:
This repository is developed from the official <a href="https://github.com/signatrix/efficientdet"> EfficientDet </a> repository.
We used the original code from <a href="https://github.com/signatrix/efficientdet"> this </a> repository, modifying some parts for the SparseSwin model.

## Datasets

| Dataset  | Classes | #Train images | #Validation images |
| -------- | :-----: | :-----------: | :----------------: |
| COCO2017 |   80    |     118k      |         5k         |

Create a data folder under the repository,

```
cd {repo_root}
mkdir data
```

-   **COCO**:
    Download the coco images and annotations from [coco website](http://cocodataset.org/#download).
    Make sure to put the files as the following structure:
    ```
    COCO
    ├── annotations
    │   ├── instances_train2017.json
    │   └── instances_val2017.json
    │── images
        ├── train2017
        └── val2017
    ```

## How to use our code

With our code, you can:

-   **Train your model** by running **python train.py**
-   **Evaluate mAP for COCO dataset** by running **python mAP_evaluation.py**
-   **Test your model for COCO dataset** by running **python test_dataset.py --pretrained_model path/to/trained_model**
-   **Test your model for video** by running **python test_video.py --pretrained_model path/to/trained_model --input path/to/input/file --output path/to/output/file**

## Experiments

We trained our model by using NVIDIA A100-SXM4-80GB.
Below is mAP (mean average precision) for COCO val2017 dataset

| Average Precision | IoU=0.50:0.95 |  area= all   | maxDets=100 | 0.214 |
| ----------------- | :-----------: | :----------: | :---------: | :---: |
| Average Precision |   IoU=0.50    |  area= all   | maxDets=100 | 0.342 |
| Average Precision |   IoU=0.75    |  area= all   | maxDets=100 | 0.224 |
| Average Precision | IoU=0.50:0.95 | area= small  | maxDets=100 | 0.033 |
| Average Precision | IoU=0.50:0.95 | area= medium | maxDets=100 | 0.239 |
| Average Precision | IoU=0.50:0.95 | area= large  | maxDets=100 | 0.379 |
| Average Recall    | IoU=0.50:0.95 |  area= all   |  maxDets=1  | 0.194 |
| Average Recall    | IoU=0.50:0.95 |  area= all   | maxDets=10  | 0.267 |
| Average Recall    | IoU=0.50:0.95 |  area= all   | maxDets=100 | 0.272 |
| Average Recall    | IoU=0.50:0.95 | area= small  | maxDets=100 | 0.031 |
| Average Recall    | IoU=0.50:0.95 | area= medium | maxDets=100 | 0.307 |
| Average Recall    | IoU=0.50:0.95 | area= large  | maxDets=100 | 0.478 |

## Results

Some predictions are shown below:

<img src="predictions/000000000139_prediction.jpg" width="280"> 
<img src="predictions/000000000285_prediction.jpg" width="280"> 
<img src="predictions/000000000724_prediction.jpg" width="280">

<img src="predictions/000000000802_prediction.jpg" width="280">
<img src="predictions/000000002261_prediction.jpg" width="280">
<img src="predictions/000000004495_prediction.jpg" width="280">

## Requirements

-   **python 3.6**
-   **pytorch 1.12**
-   **opencv (cv2)**
-   **tensorboard**
-   **tensorboardX** (This library could be skipped if you do not use SummaryWriter)
-   **pycocotools**

## References

    Pinasthika, K., Laksono, B.S.P., Irsal, R.B.P. and Yudistira, N., 2024. SparseSwin: Swin Transformer with Sparse Transformer Block. Neurocomputing, p.127433.

## Citation

    @article{PINASTHIKA2024127433,
        title = {SparseSwin: Swin transformer with sparse transformer block},
        journal = {Neurocomputing},
        volume = {580},
        pages = {127433},
        year = {2024},
        issn = {0925-2312},
        doi = {https://doi.org/10.1016/j.neucom.2024.127433},
        url = {https://www.sciencedirect.com/science/article/pii/S0925231224002042},
        author = {Krisna Pinasthika and Blessius Sheldo Putra Laksono and Riyandi Banovbi Putera Irsal and Syifa’ Hukma Shabiyya and Novanto Yudistira},
        keywords = {CIFAR10, CIFAR100, Computer vision, Image classification, ImageNet100, Transformer},
        abstract = {Advancements in computer vision research have put transformer architecture as the state-of-the-art in computer vision tasks. One of the known drawbacks of the transformer architecture is the high number of parameters, this can lead to a more complex and inefficient algorithm. This paper aims to reduce the number of parameters and in turn, made the transformer more efficient. We present Sparse Transformer (SparTa) Block, a modified transformer block with an addition of a sparse token converter that reduces the dimension of high-level features to the number of latent tokens. We implemented the SparTa Block within the Swin-T architecture (SparseSwin) to leverage Swin's proficiency in extracting low-level features and enhance its capability to extract information from high-level features while reducing the number of parameters. The proposed SparseSwin model outperforms other state-of-the-art models in image classification with an accuracy of 87.26%, 97.43%, and 85.35% on the ImageNet100, CIFAR10, and CIFAR100 datasets respectively. Despite its fewer parameters, the result highlights the potential of a transformer architecture using a sparse token converter with a limited number of tokens to optimize the use of the transformer and improve its performance. The code is available at https://github.com/KrisnaPinasthika/SparseSwin.}
    }
