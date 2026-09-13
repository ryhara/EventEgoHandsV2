import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


def dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
):
    if input.dim() == 4:
        assert input.size(1) == 1
        input = input.squeeze(1)
        target = target.squeeze(1)
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert (
        input.dim() == 3 or not reduce_batch_first
    ), f"input.dim()={input.dim()}, input.shape={input.shape}"

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
):
    # Average of Dice coefficient for all classes
    return dice_coeff(
        input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon
    )


# Reference: https://github.com/milesial/Pytorch-UNet/blob/master/utils/dice_score.py
def dice_loss_fn(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


# Reference: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=1.0, gamma=2.0):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        bce = F.binary_cross_entropy(inputs, targets, reduction="mean")
        bce_exp = torch.exp(-bce)
        focal_loss = alpha * (1 - bce_exp) ** gamma * bce
        return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)
        # inputs = (inputs > 0.1).float()

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class MaskLoss(nn.Module):
    def __init__(
        self,
        mode = "focal",
        multiclass: bool = False,
        alpha: float = 0.3,
        focal_gamma: float = 2.0,
        focal_alpha: float = 1.0,
    ):
        super().__init__()
        self.mode = mode
        self.multiclass = multiclass
        self.alpha = alpha
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.focal_fn = FocalLoss()
        self.dice_fn = DiceLoss()

    def forward(self, input: Tensor, target: Tensor):
        if self.mode == "focal":
            focal_loss = self.focal_fn(input, target, self.focal_alpha, self.focal_gamma)
            dice_loss = self.dice_fn(input, target)
            total_loss = (1 - self.alpha) * focal_loss + self.alpha * dice_loss
            losses = {
                "focal_loss": focal_loss,
                "dice_loss": dice_loss,
                "loss": total_loss,
            }
        else:
            bce_loss = self.bce_fn(input, target)
            dice_loss = self.dice_fn(input, target)
            total_loss = (1 - self.alpha) * bce_loss + self.alpha * dice_loss
            losses = {
                "bce_loss": bce_loss,
                "dice_loss": dice_loss,
                "loss": total_loss,
            }

        return losses
