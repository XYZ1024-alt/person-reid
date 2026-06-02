import torch
import torch.nn.functional as F


def extract_feature(predictor, cropped_img):
    output = predictor(cropped_img)

    if isinstance(output, torch.Tensor):
        return output.detach().cpu().flatten()

    return torch.as_tensor(output).detach().cpu().flatten()


def compute_similarity(feat1, feat2):
    feat1 = feat1.flatten()
    feat2 = feat2.flatten()
    return F.cosine_similarity(feat1, feat2, dim=0).item()
