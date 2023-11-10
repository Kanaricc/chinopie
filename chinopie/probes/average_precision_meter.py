# sample execution (requires torchvision)
import copy
from typing import List, Optional
import torch
import numpy as np
import math
import warnings
from torch.functional import Tensor

from .. import iddp as dist
class AveragePrecisionMeter:
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, difficult_examples=False):
        super(AveragePrecisionMeter, self).__init__()
        if dist.is_preferred():
            warnings.warn("AP Meter may not work properly with DDP. Do not trust the results if DDP sampler is used for dataset!")

        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())
        self.filenames = []

    def add(self, output: Tensor, target: Tensor, filename: List[str]):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        assert output.dtype == torch.float, "wrong output type"

        if output.dim() == 1:
            output = output.view(1, -1)
        else:
            assert (
                output.dim() == 2
            ), "wrong output size (should be 1D or 2D with one column \
                per class)"
        if target.dim() == 1:
            target = target.view(1, -1)
        else:
            assert (
                target.dim() == 2
            ), "wrong target size (should be 1D or 2D with one column \
                per class)"
        assert output.size(1)==target.size(1), "#labels not matched between output and target"
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(
                1
            ), "dimensions for output should match previously added examples."

        # store scores and targets
        # this is indeed expensive in memory. we may have more elegant way.
        self.scores = torch.cat([self.scores, output.detach().cpu()], dim=0)
        self.targets = torch.cat([self.targets, target.detach().cpu()], dim=0)

        self.filenames += copy.deepcopy(filename)  # record filenames

    def value(self, retain_topN:Optional[int]=None):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        assert self.scores.numel() != 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(
                scores, targets, self.difficult_examples,retain_topN
            )
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True,retain_topN:Optional[int]=None):
        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)
        if retain_topN is not None:
            indices=indices[:retain_topN]

        # Computes prec@i
        pos_count = 0.0
        total_count = 0.0
        precision_at_i = 0.0
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        if pos_count != 0:
            precision_at_i /= pos_count
            return precision_at_i
        else:
            return 0

    def overall(self, threshold=0.0):
        assert self.scores.numel() != 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.clone().cpu().numpy()
        targets[targets == -1] = 0
        return self.evaluation(scores, targets, threshold)

    def overall_topk(self, k, threshold=0.0):
        targets = self.targets.clone().cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= threshold else -1
        return self.evaluation(scores, targets, threshold)

    def evaluation(self, scores_, targets_, threshold=0.0):
        n, n_class = scores_.shape
        # Nc: number of correct positive prediction, true postive
        # Np: number of inputs predicted as positive, true positive + false positive
        # Ng: number of target tags as positive, true positive + false negative
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            Ng[k] = np.sum(targets == 1)
            Np[k] = np.sum(scores >= threshold)
            Nc[k] = np.sum(targets * (scores >= threshold))
        OP = np.sum(Nc) / np.sum(Np)
        OR = np.sum(Nc) / np.sum(Ng)
        OF1 = (2 * OP * OR) / (OP + OR)

        Np[Np==0]=1
        CP = np.sum(Nc / Np) / n_class
        CR = np.sum(Nc / Ng) / n_class
        CF1 = (2 * CP * CR) / (CP + CR)

        return OP, OR, OF1, CP, CR, CF1
