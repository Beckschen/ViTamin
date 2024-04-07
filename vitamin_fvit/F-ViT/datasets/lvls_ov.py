from mmdet.datasets.lvis import LVISV1Dataset
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose

import os.path as osp
import mmcv
import json
import warnings

import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

@DATASETS.register_module()
class LVISV1DatasetOV(LVISV1Dataset):
    def __init__(self,
                 seen_classes='datasets/lvis_v1_seen_classes.json',
                 unseen_classes='datasets/lvis_v1_unseen_classes.json',
                 all_classes='datasets/lvis_v1_all_classes.json',
                 **kwargs):
        self.seen_classes = json.load(open(seen_classes))
        self.unseen_classes = json.load(open(unseen_classes))
        super().__init__(**kwargs)
    # def __init__(self,
    #              ann_file,
    #              pipeline,
    #              seen_classes='datasets/lvis_v1_seen_classes.json',
    #              unseen_classes='datasets/lvis_v1_unseen_classes.json',
    #              all_classes='datasets/lvis_v1_all_classes.json',
    #              classes=None,
    #              data_root=None,
    #              img_prefix='',
    #              seg_prefix=None,
    #              seg_suffix='.png',
    #              proposal_file=None,
    #              test_mode=False,
    #              filter_empty_gt=True,
    #              file_client_args=dict(backend='disk')):
    #     self.ann_file = ann_file
    #     self.data_root = data_root
    #     self.img_prefix = img_prefix
    #     self.seg_prefix = seg_prefix
    #     self.seg_suffix = seg_suffix
    #     self.proposal_file = proposal_file
    #     self.test_mode = test_mode
    #     self.filter_empty_gt = filter_empty_gt
    #     self.file_client = mmcv.FileClient(**file_client_args)
    #     # self.CLASSES = self.get_classes(classes)

    #     self.seen_classes = json.load(open(seen_classes))
    #     self.unseen_classes = json.load(open(unseen_classes))
    #     self.all_classes = json.load(open(all_classes))

    #     if test_mode:
    #         self.CLASSES = self.all_classes
    #     else:
    #         self.CLASSES = self.seen_classes

    #     self.data_infos = self.load_annotations(self.ann_file)
    #     self.proposals = None

    #     name2cat = {v['name']: k for k, v in self.coco.cats.items()}
    #     self.cat_ids = [name2cat[name] for name in self.CLASSES]

    #     self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

    #     # self.seen_cat_ids = [name2cat[name] for name in seen_classes]
    #     # self.unseen_cat_ids = [name2cat[name] for name in unseen_classes]

    #     # filter images too small and containing no annotations
    #     # does not filter annotations not in self.cat_ids
    #     # _parse_ann_info() will filter annotations not in self.cat_ids
    #     if not test_mode:
    #         valid_inds = self._filter_imgs()
    #         self.data_infos = [self.data_infos[i] for i in valid_inds]
    #         if self.proposals is not None:
    #             self.proposals = [self.proposals[i] for i in valid_inds]
    #         # set group flag for the sampler
    #         self._set_group_flag()

    #     # processing pipeline
    #     self.pipeline = Compose(pipeline)

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=True,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        """Evaluation in LVIS protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None):
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls. If set to a list, the average recall of all IoUs will
                also be computed. Default: 0.5.

        Returns:
            dict[str, float]: LVIS style metrics.
        """

        try:
            import lvis
            if getattr(lvis, '__version__', '0') >= '10.5.3':
                warnings.warn(
                    'mmlvis is deprecated, please install official lvis-api by "pip install git+https://github.com/lvis-dataset/lvis-api.git"',  # noqa: E501
                    UserWarning)
            from lvis import LVISEval, LVISResults
        except ImportError:
            raise ImportError(
                'Package lvis is not installed. Please run "pip install git+https://github.com/lvis-dataset/lvis-api.git".'  # noqa: E501
            )
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError('metric {} is not supported'.format(metric))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)

        eval_results = OrderedDict()
        # get original api
        lvis_gt = self.coco
        for metric in metrics:
            msg = 'Evaluating {}...'.format(metric)
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results['AR@{}'.format(num)] = ar[i]
                    log_msg.append('\nAR@{}\t{:.4f}'.format(num, ar[i]))
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError('{} is not in results'.format(metric))
            try:
                lvis_dt = LVISResults(lvis_gt, result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            lvis_eval = LVISEval(lvis_gt, lvis_dt, iou_type)
            lvis_eval.params.imgIds = self.img_ids
            if metric == 'proposal':
                lvis_eval.params.useCats = 0
                lvis_eval.params.maxDets = list(proposal_nums)
                lvis_eval.evaluate()
                lvis_eval.accumulate()
                lvis_eval.summarize()
                for k, v in lvis_eval.get_results().items():
                    if k.startswith('AR'):
                        val = float('{:.3f}'.format(float(v)))
                        eval_results[k] = val
            else:
                lvis_eval.evaluate()
                lvis_eval.accumulate()
                lvis_eval.summarize()
                lvis_results = lvis_eval.get_results()
                if classwise and metric == 'segm':  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = lvis_eval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    results_per_category50 = []
                    results_per_category50_seen = []
                    results_per_category50_unseen = []

                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        # the dimensions of precisions are
                        # [num_thrs, num_recalls, num_cats, num_area_rngs]
                        name = self.coco.load_cats([catId])[0]['name']
                        precision = precisions[:, :, idx, 0]
                        precision = precision[precision > -1]
                        ap = np.mean(precision) if precision.size else float("nan")
                        results_per_category.append(
                            (f'{name}', f'{float(ap):0.3f}'))
                        
                        precision50 = precisions[0, :, idx, 0]
                        precision50 = precision50[precision50 > -1]
                        ap50 = np.mean(precision50) if precision50.size else float("nan")
                        results_per_category50.append( float(ap50 * 100))
                        if name in self.seen_classes:
                            results_per_category50_seen.append(float(ap50 * 100))
                        if name in self.unseen_classes:
                            results_per_category50_unseen.append(float(ap50 * 100))

                    base_ap50 = np.nanmean(results_per_category50_seen)
                    novel_ap50 = np.nanmean(results_per_category50_unseen)
                    all_ap50 = np.nanmean(results_per_category50)
                    eval_results['base_mask_ap50'] = float(f'{base_ap50:.3f}')
                    eval_results['novel_mask_ap50'] = float(f'{novel_ap50:.3f}')
                    eval_results['all_mask_ap50'] = float(f'{all_ap50:.3f}')
                for k, v in lvis_results.items():
                    if k.startswith('AP'):
                        key = '{}_{}'.format(metric, k)
                        val = float('{:.3f}'.format(float(v)))
                        eval_results[key] = val
                ap_summary = ' '.join([
                    '{}:{:.3f}'.format(k, float(v))
                    for k, v in lvis_results.items() if k.startswith('AP')
                ])
                eval_results['{}_mAP_copypaste'.format(metric)] = ap_summary
            lvis_eval.print_results()
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
