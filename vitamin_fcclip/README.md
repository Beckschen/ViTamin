# ViTamin for Open-Vocabulary Segmentation
The folder includes the implementation of ViTamin for open-vocabulary segmentation. 

We propose Sliding FC-CLIP which adapts ViTamin within the FC-CLIP framework. Thanks, FC-CLIP!

## Installation and Getting Started

Please follows FC-CLIP's [installation instructions](INSTALL.md) and [Getting Started with  FC-CLIP](GETTING_STARTED.md).

The configuration for ViTamin is provided in "./configs/coco/panoptic-segmentation/fcclip/fcclip_vitamin_l_eval_ade20k.yaml"

## 🔥 Model Zoo
<table>
<thead>
  <tr>
    <th align="center" style="text-align:center">image encoder </th>
    <th align="center" style="text-align:center" colspan="1">ADE20K(A-150)</th>
    <th align="center" style="text-align:center" colspan="1">Cityscapes</th>
    <th align="center" style="text-align:center" colspan="1">Mapillary Vistas</th>
    <th align="center" style="text-align:center">ADE20K <br> (A-150)</th>
    <th align="center" style="text-align:center">ADE20K-Full <br> (A-847)</th>
    <th align="center" style="text-align:center">Pascal Context 459 <br> (PC-459)</th>
    <th align="center" style="text-align:center">Pascal Context 59 <br> (PC-59)</th>
    <th align="center" style="text-align:center">Pascal VOC 21 <br> (PAS-21) </th>
    <th align="center" style="text-align:center">download </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center"></td>
    <td align="center">PQ</td>
    <td align="center">PQ</td>
    <td align="center">PQ</td>
    <td align="center">mIoU</td>
    <td align="center">mIoU</td>
    <td align="center">mIoU</td>
    <td align="center">mIoU</td>
    <td align="center">mIoU</td>
  </tr>
  <tr>
    <td align="center"><a href="configs/coco/panoptic-segmentation/fcclip_convnext_large_eval_ade20k.yaml"> ConvNeXt-L</a></td>
    <td align="center">26.8</td>
    <td align="center">44.0</td>
    <td align="center">18.3</td>
    <td align="center">34.1</td>
    <td align="center">14.8</td>
    <td align="center">18.2</td>
    <td align="center">58.4</td>
    <td align="center">81.8</td>
    <td align="center"><a href="https://drive.google.com/file/d/1-91PIns86vyNaL3CzMmDD39zKGnPMtvj/view?usp=sharing"> checkpoint </a></td>
  </tr>
  <tr>
    <td align="center"><a href="configs/coco/panoptic-segmentation/fcclip_convnext_large_eval_ade20k.yaml"> ViT-L/14</a></td>
    <td align="center">24.6</td>
    <td align="center">40.7</td>
    <td align="center">16.5</td>
    <td align="center">31.8</td>
    <td align="center">14.3</td>
    <td align="center">18.3</td>
    <td align="center">55.1</td>
    <td align="center">81.5</td>
    <td align="center"><a href="https://drive.google.com"> checkpoint </a></td>
  </tr>
  <tr>
    <td align="center"><a href="configs/coco/panoptic-segmentation/fcclip_vitamin_l_eval_ade20k.yaml"> ViTamin-L-</a></td>
    <td align="center">27.3</td>
    <td align="center">44.0</td>
    <td align="center">18.2</td>
    <td align="center">35.6</td>
    <td align="center">16.1</td>
    <td align="center">20.4</td>
    <td align="center">58.4</td>
    <td align="center">83.4</td>
    <td align="center"><a href="https://drive.google.com"> checkpoint </a></td>
  </tr>

</tbody>
</table>

## Citing ViTamin

```
@inproceedings{chen2024vitamin,
  title={ViTamin: Design Scalable Vision Models in the Vision-language Era},
  author={Chen, Jieneng and Yu, Qihang and Shen, Xiaohui and Yuille, ALan and Chen, Liang-Chieh},
  journal={arXiv preprint arXiv:xxx.xxxxx},
  year={2024}
}
```

--------------------------


## Original FC-CLIP README

This repo contains thr code for our paper **Convolutions Die Hard: Open-Vocabulary Panoptic Segmentation with Single Frozen Convolutional CLIP**

<div align="center">
  <img src="imgs/teaser.png" width="100%" height="100%"/>
</div><br/>

*FC-CLIP* is an universal model for open-vocabulary image segmentation problems, consisting of a class-agnostic segmenter, in-vocabulary classifier, out-of-vocabulary classifier. With everything built upon a shared single frozen convolutional CLIP model, *FC-CLIP* not only achieves state-of-the-art performance on various open-vocabulary segmentation benchmarks, but also enjoys a much lower training (3.2 days with 8 V100) and testing costs compared to prior arts.


## Installation

See [installation instructions](INSTALL.md).

## Getting Started

See [Preparing Datasets for FC-CLIP](datasets/README.md).

See [Getting Started with  FC-CLIP](GETTING_STARTED.md).

We also support FC-CLIP with [HuggingFace 🤗 Demo](https://huggingface.co/spaces/fun-research/FC-CLIP)

## Model Zoo

<table>
<thead>
  <tr>
    <th align="center"></th>
    <th align="center" style="text-align:center" colspan="3"><a href="logs/testing/ade20k.log">ADE20K(A-150)</th>
    <th align="center" style="text-align:center" colspan="3"><a href="logs/testing/cityscapes.log">Cityscapes</th>
    <th align="center" style="text-align:center" colspan="2"><a href="logs/testing/mapillary_vistas.log">Mapillary Vistas</th>
    <th align="center" style="text-align:center"><a href="logs/testing/a-847.log">ADE20K-Full <br> (A-847)</th>
    <th align="center" style="text-align:center"><a href="logs/testing/pc-59.log">Pascal Context 59 <br> (PC-59)</th>
    <th align="center" style="text-align:center"><a href="logs/testing/pc-459.log">Pascal Context 459 <br> (PC-459)</th>
    <th align="center" style="text-align:center"><a href="logs/testing/pc-21.log">Pascal VOC 21 <br> (PAS-21) </th>
    <th align="center" style="text-align:center"><a href="logs/testing/pc-20.log">Pascal VOC 20 <br> (PAS-20) </th>
    <th align="center" style="text-align:center" colspan="3"><a href="logs/testing/coco.log">COCO <br> (training dataset)</th>
    <th align="center" style="text-align:center">download </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center"></td>
    <td align="center">PQ</td>
    <td align="center">mAP</td>
    <td align="center">mIoU</td>
    <td align="center">PQ</td>
    <td align="center">mAP</td>
    <td align="center">mIoU</td>
    <td align="center">PQ</td>
    <td align="center">mIoU</td>
    <td align="center">mIoU</td>
    <td align="center">mIoU</td>
    <td align="center">mIoU</td>
    <td align="center">mIoU</td>
    <td align="center">mIoU</td>
    <td align="center">PQ</td>
    <td align="center">mAP</td>
    <td align="center">mIoU</td>
  </tr>
  <tr>
    <td align="center"><a href="configs/coco/panoptic-segmentation/fcclip_convnext_large_eval_ade20k.yaml"> FC-CLIP </a></td>
    <td align="center">26.8</td>
    <td align="center">16.8</td>
    <td align="center">34.1</td>
    <td align="center">44.0</td>
    <td align="center">26.8</td>
    <td align="center">56.2</td>
    <td align="center">18.3</td>
    <td align="center">27.8</td>
    <td align="center">14.8</td>
    <td align="center">58.4</td>
    <td align="center">18.2</td>
    <td align="center">81.8</td>
    <td align="center">95.4</td>
    <td align="center">54.4</td>
    <td align="center">44.6</td>
    <td align="center">63.7</td>
    <td align="center"><a href="https://drive.google.com/file/d/1-91PIns86vyNaL3CzMmDD39zKGnPMtvj/view?usp=sharing"> checkpoint </a></td>
  </tr>
</tbody>
</table>

## <a name="Citing FC-CLIP"></a>Citing  FC-CLIP

If you use FC-CLIP in your research, please use the following BibTeX entry.

```BibTeX
@inproceedings{yu2023fcclip,
  title={Convolutions Die Hard: Open-Vocabulary Panoptic Segmentation with Single Frozen Convolutional CLIP},
  author={Qihang Yu and Ju He and Xueqing Deng and Xiaohui Shen and Liang-Chieh Chen},
  journal={arXiv},
  year={2023}
}
```

## Acknowledgement

Mask2Former (https://github.com/facebookresearch/Mask2Former)

ODISE (https://github.com/NVlabs/ODISE)

