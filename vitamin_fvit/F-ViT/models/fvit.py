from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import TwoStageDetector


@DETECTORS.register_module()
class FViT(TwoStageDetector):
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        # num_seen_classes = self.roi_head.bbox_head.num_classes
        # num_classes = self.roi_head.bbox_head.all_embed.shape[1] - 1
        # self.roi_head.bbox_head.num_classes = num_classes
        # self.roi_head.mask_head.num_classes = num_classes
        assert self.with_bbox, 'Bbox head must be implemented.'
        mlvl_feats = self.backbone(img)
        if self.with_neck:
            x = self.neck(mlvl_feats[:-1])
        else:
            x = mlvl_feats[:-1]
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        res = self.roi_head.simple_test(x,
                                        proposal_list,
                                        img_metas,
                                        vlm_feat=mlvl_feats[-1],
                                        rescale=rescale)

        return res

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      gt_captions=None,
                      gt_embeds=None,
                      **kwargs):
        res_feats = self.backbone(img)
        if self.with_neck:
            x = self.neck(res_feats[:-1])
        else:
            x = res_feats[:-1]

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 res_feats=res_feats,
                                                 gt_captions=gt_captions,
                                                 gt_embeds=gt_embeds,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses
