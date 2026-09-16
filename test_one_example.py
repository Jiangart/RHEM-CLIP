import os
import cv2
import torch
import numpy as np
import argparse
from PIL import Image
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

import RHEMCLIP_lib
from prompt_ensemble import RHEMCLIP_PromptLearner
from utils import get_transform

def save_heatmap(anomaly_map, orig_img_path, save_path, size=518):
    
    img = cv2.imread(orig_img_path)
    img = cv2.resize(img, (size, size))
    v_max = anomaly_map.max()
    v_min = anomaly_map.min()
    amap = (anomaly_map - v_min) / (v_max - v_min + 1e-8)

    gamma = 2.0
    amap = np.power(amap, gamma)

    amap = (amap * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(amap, cv2.COLORMAP_JET)

    vis = cv2.addWeighted(img, 0.4, heatmap, 0.6, 0)

    cv2.imwrite(save_path, vis)
def anomaly_aware_feature_smoothing(scores, feats, k=5, alpha=0.7, tau=0.2, use_max=False):
    B, N, C = feats.shape
    feats = F.normalize(feats, dim=-1)
    sim = torch.bmm(feats, feats.transpose(1, 2))
    topk_sim, topk_idx = torch.topk(sim, k=k, dim=-1)
    expanded_scores = scores.unsqueeze(1).expand(-1, N, -1)
    neighbor_scores = torch.gather(expanded_scores, 2, topk_idx)

    gate = torch.sigmoid((neighbor_scores - scores.unsqueeze(-1)) / tau) if use_max else 1.0
    weights = F.softmax(topk_sim * gate, dim=-1)
    refined = torch.sum(weights * neighbor_scores, dim=-1)
    final = alpha * scores + (1 - alpha) * refined
    return torch.max(final, scores) if use_max else final

def compute_spatial_gradient(anomaly_map):
    if len(anomaly_map.shape) == 2: anomaly_map = anomaly_map.unsqueeze(0).unsqueeze(0)
    if len(anomaly_map.shape) == 3: anomaly_map = anomaly_map.unsqueeze(1)

    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=anomaly_map.dtype, device=anomaly_map.device).view(1,1,3,3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=anomaly_map.dtype, device=anomaly_map.device).view(1,1,3,3)

    grad_x = F.conv2d(anomaly_map, kx, padding=1)
    grad_y = F.conv2d(anomaly_map, ky, padding=1)
    gradient = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    return (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-8)

def compute_layer_disagreement(stack_maps):
    disagreement = torch.std(stack_maps, dim=1)
    return (disagreement - disagreement.min()) / (disagreement.max() - disagreement.min() + 1e-8)
def inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    RHEMCLIP_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx
    }
    model, _ = RHEMCLIP_lib.load("ViT-L/14@336px", device=device, design_details=RHEMCLIP_parameters)
    model.visual.DAPM_replace(DPAM_layer=20)

    prompt_learner = RHEMCLIP_PromptLearner(model.to("cpu"), RHEMCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])

    model.to(device)
    prompt_learner.to(device)
    model.eval()

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    preprocess, _ = get_transform(args)
    orig_image = Image.open(args.image_path).convert('RGB')
    image = preprocess(orig_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)

        layer_sim_maps = []
        active_layer_ids = []

        for idx, patch_feature in enumerate(patch_features):
            if idx >= args.feature_map_layer[0]:
                patch_feature /= patch_feature.norm(dim=-1, keepdim=True)
                similarity, _ = RHEMCLIP_lib.compute_similarity(patch_feature, text_features[0])

                current_layer_id = args.features_list[idx]

                if args.enable_smoothing and (current_layer_id in args.smooth_layers):
                    refined_scores = anomaly_aware_feature_smoothing(
                        scores=similarity[:, 1:, 1], feats=patch_feature[:, 1:, :],
                        k=args.smooth_k, alpha=args.smooth_alpha, use_max=args.smooth_use_max
                    )
                    similarity[:, 1:, 1] = refined_scores

                sim_map = RHEMCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
                layer_sim_maps.append(sim_map)
                active_layer_ids.append(current_layer_id)

        stack_maps = torch.stack(layer_sim_maps, dim=1) # [1, Layers, 2, H, W]
        baseline_mean = torch.mean(stack_maps, dim=1, keepdim=True)
        dist = (stack_maps - baseline_mean).pow(2).mean(dim=[-2, -1], keepdim=True)
        layer_weights = F.softmax(-dist * 5.0, dim=1)

        boost_list = [args.boost_ratio if lid == 18 else 1.0 for lid in active_layer_ids]
        boost_factors = torch.tensor(boost_list, device=device).view(1, -1, 1, 1, 1)
        layer_weights = (layer_weights * boost_factors) / (layer_weights * boost_factors).sum(dim=1, keepdim=True)
        fused_map = torch.sum(stack_maps * layer_weights, dim=1) # [1, 2, H, W]
        anomaly_tensor = fused_map[:, 1, :, :]

        if args.enable_disagreement:
            dis_map = compute_layer_disagreement(stack_maps)[:, 1, :, :]
            anomaly_tensor += args.disagreement_weight * dis_map

        if args.enable_gradient:
            if 18 in active_layer_ids:
                idx_18 = active_layer_ids.index(18)
                grad_map = compute_spatial_gradient(stack_maps[:, idx_18, 1, :, :])
                anomaly_tensor += args.gradient_weight * grad_map.squeeze(1)

        anomaly_map = anomaly_tensor.cpu().numpy()[0]
        anomaly_map = gaussian_filter(anomaly_map, sigma=args.sigma)

        save_name = os.path.basename(args.image_path).split('.')[0] + "_result.png"
        out_path = os.path.join(args.save_path, save_name)
        os.makedirs(args.save_path, exist_ok=True)

        save_heatmap(anomaly_map, args.image_path, out_path, size=args.image_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("RHEMCLIP Single Image Inference")
    parser.add_argument("--image_path", type=str, required=True, help="Image path")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Model weight path")
    parser.add_argument("--save_path", type=str, default='./output_vis')

    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=0)
    parser.add_argument("--sigma", type=int, default=4)

    parser.add_argument("--enable_smoothing", action="store_true")
    parser.add_argument("--smooth_layers", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--smooth_k", type=int, default=5)
    parser.add_argument("--smooth_alpha", type=float, default=0.7)
    parser.add_argument("--smooth_use_max", action="store_true")

    parser.add_argument("--boost_ratio", type=float, default=3)
    parser.add_argument("--enable_gradient", action="store_true")
    parser.add_argument("--gradient_weight", type=float, default=0.65)
    parser.add_argument("--enable_disagreement", action="store_true")
    parser.add_argument("--disagreement_weight", type=float, default=0.3)

    args = parser.parse_args()
    inference(args)
