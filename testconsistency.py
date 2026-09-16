import RHEMCLIP_lib
import torch
import torch.nn as nn
import torch.nn.functional as F
from prompt_ensemble import RHEMCLIP_PromptLearner
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import numpy as np
import argparse
from tabulate import tabulate
from metrics import image_level_metrics, pixel_level_metrics
from scipy.ndimage import gaussian_filter
from utils import get_transform
from sklearn.metrics import precision_recall_curve


class LayerSpecializedAnomalyPromptLearner(nn.Module):
    def __init__(self, base_prompt_learner, num_layers=4, init_std=0.01):
        super().__init__()
        self.base_prompt_learner = base_prompt_learner
        self.num_layers = num_layers

        ctx_pos_shape = self.base_prompt_learner.ctx_pos.shape
        ctx_neg_shape = self.base_prompt_learner.ctx_neg.shape

        self.layer_ctx_pos_offsets = nn.Parameter(
            torch.zeros(num_layers, *ctx_pos_shape)
        )
        self.layer_ctx_neg_offsets = nn.Parameter(
            torch.zeros(num_layers, *ctx_neg_shape)
        )

        nn.init.normal_(self.layer_ctx_pos_offsets, std=init_std)
        nn.init.normal_(self.layer_ctx_neg_offsets, std=init_std)

    def _build_prompts_from_ctx(self, ctx_pos, ctx_neg):
        prefix_pos = self.base_prompt_learner.token_prefix_pos
        prefix_neg = self.base_prompt_learner.token_prefix_neg
        suffix_pos = self.base_prompt_learner.token_suffix_pos
        suffix_neg = self.base_prompt_learner.token_suffix_neg

        prompts_pos = torch.cat([prefix_pos, ctx_pos, suffix_pos], dim=2)
        prompts_neg = torch.cat([prefix_neg, ctx_neg, suffix_neg], dim=2)

        _, _, l, d = prompts_pos.shape
        prompts_pos = prompts_pos.reshape(-1, l, d)

        _, _, l, d = prompts_neg.shape
        prompts_neg = prompts_neg.reshape(-1, l, d)

        prompts = torch.cat([prompts_pos, prompts_neg], dim=0)

        _, _, d_tok = self.base_prompt_learner.tokenized_prompts_pos.shape
        tokenized_prompts_pos = self.base_prompt_learner.tokenized_prompts_pos.reshape(-1, d_tok)
        tokenized_prompts_neg = self.base_prompt_learner.tokenized_prompts_neg.reshape(-1, d_tok)
        tokenized_prompts = torch.cat((tokenized_prompts_pos, tokenized_prompts_neg), dim=0)

        return prompts, tokenized_prompts, self.base_prompt_learner.compound_prompts_text

    def forward(self, cls_id=None):
        base_ctx_pos = self.base_prompt_learner.ctx_pos
        base_ctx_neg = self.base_prompt_learner.ctx_neg

        base_prompts, tokenized_prompts, compound_prompts_text = self._build_prompts_from_ctx(
            base_ctx_pos, base_ctx_neg
        )

        layer_prompts_list = []
        for l in range(self.num_layers):
            ctx_pos_l = base_ctx_pos + self.layer_ctx_pos_offsets[l]
            ctx_neg_l = base_ctx_neg + self.layer_ctx_neg_offsets[l]
            prompts_l, _, _ = self._build_prompts_from_ctx(ctx_pos_l, ctx_neg_l)
            layer_prompts_list.append(prompts_l)

        return base_prompts, layer_prompts_list, tokenized_prompts, compound_prompts_text


class LearnableReliabilityEstimator(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=32):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, sim_map):
        x = self.pool(sim_map)
        logit = self.mlp(x)
        return logit


def anomaly_aware_feature_smoothing(
        scores,
        feats,
        k=5,
        alpha=0.7,
        tau=0.2,
        use_max=False
):
    B, N, C = feats.shape
    feats = F.normalize(feats, dim=-1)
    sim = torch.bmm(feats, feats.transpose(1, 2))
    topk_sim, topk_idx = torch.topk(sim, k=k, dim=-1)
    expanded_scores = scores.unsqueeze(1).expand(-1, N, -1)
    neighbor_scores = torch.gather(expanded_scores, 2, topk_idx)

    if use_max:
        score_diff = neighbor_scores - scores.unsqueeze(-1)
        gate = torch.sigmoid(score_diff / tau)
    else:
        gate = 1.0

    weights = F.softmax(topk_sim * gate, dim=-1)
    refined = torch.sum(weights * neighbor_scores, dim=-1)
    final = alpha * scores + (1 - alpha) * refined
    if use_max:
        final = torch.max(final, scores)
    return final


def compute_spatial_gradient(anomaly_map):
    if len(anomaly_map.shape) == 3:
        anomaly_map = anomaly_map.unsqueeze(1)

    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=anomaly_map.dtype, device=anomaly_map.device
    ).view(1, 1, 3, 3)

    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=anomaly_map.dtype, device=anomaly_map.device
    ).view(1, 1, 3, 3)

    grad_x = F.conv2d(anomaly_map, kx, padding=1)
    grad_y = F.conv2d(anomaly_map, ky, padding=1)

    gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-8)
    return gradient.squeeze(1)


def compute_layer_disagreement(stack_maps):
    disagreement = torch.std(stack_maps, dim=1)
    disagreement = (disagreement - disagreement.min()) / (disagreement.max() - disagreement.min() + 1e-8)
    return disagreement


def _to_numpy_1d(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().reshape(-1)
    if isinstance(x, list):
        return np.asarray([v.item() if torch.is_tensor(v) else v for v in x]).reshape(-1)
    return np.asarray(x).reshape(-1)


def compute_max_f1(gt, scores):
    gt = _to_numpy_1d(gt).astype(np.uint8)
    scores = _to_numpy_1d(scores).astype(np.float32)

    precision, recall, thresholds = precision_recall_curve(gt, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    best_idx = np.nanargmax(f1)
    best_f1 = float(f1[best_idx])

    if len(thresholds) == 0:
        best_thr = 0.5
    else:
        best_thr = float(thresholds[min(best_idx, len(thresholds) - 1)])

    return best_f1, best_thr


def test(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess, target_transform = get_transform(args)

    test_data = Dataset(
        root=args.data_path,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=args.dataset
    )
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list
    logger = get_logger(args.save_path)

    RHEMCLIP_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx
    }

    model, _ = RHEMCLIP_lib.load(
        "ViT-L/14@336px",
        device=device,
        design_details=RHEMCLIP_parameters
    )

    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # -------------------------
    # Prompt learner
    # -------------------------
    if args.use_layer_specialized_prompt:
        base_prompt_learner = RHEMCLIP_PromptLearner(model.to("cpu"), RHEMCLIP_parameters)
        base_prompt_learner.to(device)

        prompt_learner = LayerSpecializedAnomalyPromptLearner(
            base_prompt_learner=base_prompt_learner,
            num_layers=len(args.features_list),
            init_std=args.layer_prompt_init_std
        ).to(device)
        logger.info("Using layer-specialized prompts for testing.")
    else:
        prompt_learner = RHEMCLIP_PromptLearner(model.to("cpu"), RHEMCLIP_parameters)
        prompt_learner.to(device)
        logger.info("Using shared prompts for testing.")

    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)


    if args.use_reliability_fusion:
        reliability_estimator = LearnableReliabilityEstimator(
            in_channels=2,
            hidden_dim=args.reliability_hidden_dim
        ).to(device)

        if "reliability_estimator" not in checkpoint:
            raise KeyError("Checkpoint does not contain 'reliability_estimator'.")

        reliability_estimator.load_state_dict(checkpoint["reliability_estimator"])
        reliability_estimator.eval()
    else:
        reliability_estimator = None

    model.to(device)

    if args.use_layer_specialized_prompt:
        base_prompts, layer_prompts_list, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)

        text_features_global = model.encode_text_learn(
            base_prompts, tokenized_prompts, compound_prompts_text
        ).float()
        text_features_global = torch.stack(torch.chunk(text_features_global, dim=0, chunks=2), dim=1)
        text_features_global = text_features_global / text_features_global.norm(dim=-1, keepdim=True)

        layer_text_features = []
        for lp in layer_prompts_list:
            tf = model.encode_text_learn(lp, tokenized_prompts, compound_prompts_text).float()
            tf = torch.stack(torch.chunk(tf, dim=0, chunks=2), dim=1)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            layer_text_features.append(tf)
    else:
        prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
        text_features_global = model.encode_text_learn(
            prompts, tokenized_prompts, compound_prompts_text
        ).float()
        text_features_global = torch.stack(torch.chunk(text_features_global, dim=0, chunks=2), dim=1)
        text_features_global = text_features_global / text_features_global.norm(dim=-1, keepdim=True)

    results = {obj: {'gt_sp': [], 'pr_sp': [], 'imgs_masks': [], 'anomaly_maps': []} for obj in obj_list}

    model.eval()
    with torch.no_grad():
        for items in tqdm(test_dataloader):
            image = items['img'].to(device)
            cls_name = items['cls_name'][0]

            image_features, patch_features = model.encode_image(
                image, args.features_list
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_probs = (image_features @ text_features_global.permute(0, 2, 1) / 0.07).softmax(-1)
            results[cls_name]['pr_sp'].extend(text_probs[:, 0, 1].cpu())
            results[cls_name]['gt_sp'].extend(items['anomaly'].cpu())

            gt_mask = items['img_mask'].detach().cpu()
            gt_mask = (gt_mask > 0.5).to(torch.uint8)
            results[cls_name]['imgs_masks'].append(gt_mask)

            layer_sim_maps = []
            active_layer_ids = []
            selected_layer_idx = 0

            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)

                    if args.use_layer_specialized_prompt:
                        text_feature_for_this_layer = layer_text_features[selected_layer_idx][0]
                    else:
                        text_feature_for_this_layer = text_features_global[0]

                    similarity, _ = RHEMCLIP_lib.compute_similarity(
                        patch_feature, text_feature_for_this_layer
                    )

                    current_layer_id = args.features_list[idx]

                    if args.enable_smoothing and (current_layer_id in args.smooth_layers):
                        raw_scores = similarity[:, 1:, 1]
                        patch_feats = patch_feature[:, 1:, :]
                        refined_scores = anomaly_aware_feature_smoothing(
                            scores=raw_scores,
                            feats=patch_feats,
                            k=args.smooth_k,
                            alpha=args.smooth_alpha,
                            tau=args.smooth_tau,
                            use_max=args.smooth_use_max
                        )
                        similarity[:, 1:, 1] = refined_scores

                    sim_map = RHEMCLIP_lib.get_similarity_map(
                        similarity[:, 1:, :], args.image_size
                    ).permute(0, 3, 1, 2)

                    layer_sim_maps.append(sim_map)
                    active_layer_ids.append(current_layer_id)
                    selected_layer_idx += 1

            stack_maps = torch.stack(layer_sim_maps, dim=1)  # [B, L, 2, H, W]

    
            if args.use_reliability_fusion:
                reliability_logits = []
                for i in range(stack_maps.shape[1]):
                    l_map = stack_maps[:, i, ...]
                    l_logit = reliability_estimator(l_map)
                    reliability_logits.append(l_logit)

                reliability_logits = torch.stack(reliability_logits, dim=1)  # [B, L, 1]
                layer_weights = F.softmax(reliability_logits, dim=1).unsqueeze(-1).unsqueeze(-1)
                fused_map = torch.sum(stack_maps * layer_weights, dim=1)  # [B, 2, H, W]
            else:
                fused_map = torch.mean(stack_maps, dim=1)
         
            anomaly_tensor = fused_map[:, 1, :, :]

            if args.enable_gradient:
                target_grad_layer = 18
                if target_grad_layer in active_layer_ids:
                    idx_18 = active_layer_ids.index(target_grad_layer)
                    layer_18_map = stack_maps[:, idx_18, 1, :, :]
                    gradient_map = compute_spatial_gradient(layer_18_map)
                else:
                    gradient_map = compute_spatial_gradient(anomaly_tensor)

                anomaly_tensor = anomaly_tensor + args.gradient_weight * gradient_map

            anomaly_map = anomaly_tensor.cpu().numpy()
            anomaly_map = np.stack([gaussian_filter(m, sigma=args.sigma) for m in anomaly_map])
            results[cls_name]['anomaly_maps'].append(torch.from_numpy(anomaly_map))

    table_ls = []
    image_auroc_list, image_ap_list, image_maxf1_list = [], [], []
    pixel_auroc_list, pixel_aupro_list, pixel_maxf1_list = [], [], []

    for obj in obj_list:
        table = [obj]
        results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks']).squeeze().numpy()
        results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).numpy()

        if 'pixel' in args.metrics:
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            pixel_maxf1, _ = compute_max_f1(
                results[obj]['imgs_masks'],
                results[obj]['anomaly_maps']
            )

            table.extend([
                str(np.round(pixel_auroc * 100, 1)),
                str(np.round(pixel_aupro * 100, 1)),
                str(np.round(pixel_maxf1 * 100, 1))
            ])

            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
            pixel_maxf1_list.append(pixel_maxf1)

        if 'image' in args.metrics:
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            image_maxf1, _ = compute_max_f1(
                results[obj]['gt_sp'],
                results[obj]['pr_sp']
            )

            table.extend([
                str(np.round(image_auroc * 100, 1)),
                str(np.round(image_ap * 100, 1)),
                str(np.round(image_maxf1 * 100, 1))
            ])

            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
            image_maxf1_list.append(image_maxf1)

        table_ls.append(table)

    headers = ['objects']
    if 'pixel' in args.metrics:
        headers.extend(['pixel_auroc', 'pixel_aupro', 'pixel_maxf1'])
    if 'image' in args.metrics:
        headers.extend(['image_auroc', 'image_ap', 'image_maxf1'])

    mean_row = ['mean']
    if pixel_auroc_list:
        mean_row.extend([
            str(np.round(np.mean(pixel_auroc_list) * 100, 1)),
            str(np.round(np.mean(pixel_aupro_list) * 100, 1)),
            str(np.round(np.mean(pixel_maxf1_list) * 100, 1))
        ])
    if image_auroc_list:
        mean_row.extend([
            str(np.round(np.mean(image_auroc_list) * 100, 1)),
            str(np.round(np.mean(image_ap_list) * 100, 1)),
            str(np.round(np.mean(image_maxf1_list) * 100, 1))
        ])
    table_ls.append(mean_row)

    logger.info(args.checkpoint_path)
    print(args.checkpoint_path)
    report = tabulate(table_ls, headers=headers, tablefmt="pipe")
    logger.info("\n%s", report)
    print("Inference finished.")
    logger.info("Inference finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("RHEMCLIP Test")
    parser.add_argument("--data_path", type=str, default="./data/mvtec")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--sigma", type=int, default=4)
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=0)
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--save_path", type=str, default='./results')
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--use_reliability_fusion", action="store_true")
    parser.add_argument("--use_layer_specialized_prompt", action="store_true")
    parser.add_argument("--layer_prompt_init_std", type=float, default=0.01)

    # learnable RAMF
    parser.add_argument("--reliability_hidden_dim", type=int, default=32)

    parser.add_argument("--enable_smoothing", action="store_true")
    parser.add_argument("--smooth_k", type=int, default=5)
    parser.add_argument("--smooth_alpha", type=float, default=0.7)
    parser.add_argument("--smooth_tau", type=float, default=0.2)
    parser.add_argument("--smooth_use_max", action="store_true")
    parser.add_argument("--smooth_layers", type=int, nargs="+", default=[6, 12, 18, 24])

    parser.add_argument("--enable_gradient", action="store_true")
    parser.add_argument("--gradient_weight", type=float, default=0.35)

    args = parser.parse_args()
    test(args)
