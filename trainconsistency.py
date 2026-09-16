import os
import math
import random
import argparse

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import RHEMCLIP_lib
from prompt_ensemble import RHEMCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize, get_transform
from dataset import Dataset
from logger import get_logger

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LayerSpecializedAnomalyPromptLearner(nn.Module):
    """
    Wrapper for your RHEMCLIP_PromptLearner.

    It keeps the shared/base prompt learner and adds tiny layer-specific
    residual offsets to ctx_pos / ctx_neg for each selected layer.
    """
    def __init__(self, base_prompt_learner, num_layers=4, init_std=0.01):
        super().__init__()
        self.base_prompt_learner = base_prompt_learner
        self.num_layers = num_layers

        ctx_pos_shape = self.base_prompt_learner.ctx_pos.shape
        ctx_neg_shape = self.base_prompt_learner.ctx_neg.shape

        # [L, n_cls, normal_num, n_ctx, ctx_dim]
        self.layer_ctx_pos_offsets = nn.Parameter(
            torch.zeros(num_layers, *ctx_pos_shape)
        )
        # [L, n_cls, anomaly_num, n_ctx, ctx_dim]
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

        prompts_pos = torch.cat(
            [prefix_pos, ctx_pos, suffix_pos],
            dim=2,
        )
        prompts_neg = torch.cat(
            [prefix_neg, ctx_neg, suffix_neg],
            dim=2,
        )

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

        # shared prompts for global image branch
        base_prompts, tokenized_prompts, compound_prompts_text = self._build_prompts_from_ctx(
            base_ctx_pos, base_ctx_neg
        )

        # layer-specific prompts for local branch
        layer_prompts_list = []
        for l in range(self.num_layers):
            ctx_pos_l = base_ctx_pos + self.layer_ctx_pos_offsets[l]
            ctx_neg_l = base_ctx_neg + self.layer_ctx_neg_offsets[l]
            prompts_l, _, _ = self._build_prompts_from_ctx(ctx_pos_l, ctx_neg_l)
            layer_prompts_list.append(prompts_l)

        return base_prompts, layer_prompts_list, tokenized_prompts, compound_prompts_text


class LearnableReliabilityEstimator(nn.Module):
    """
    Input: layer-wise anomaly map [B, 2, H, W]
    Output: reliability logit [B, 1]
    """
    def __init__(self, in_channels=2, hidden_dim=32):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),              # [B, 2]
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, sim_map):
        x = self.pool(sim_map)         # [B, 2, 1, 1]
        logit = self.mlp(x)            # [B, 1]
        return logit


def train(args):
    logger = get_logger(args.save_path)
    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_path, exist_ok=True)

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
    model.eval()

    train_data = Dataset(
        root=args.train_data_path,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=args.dataset
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True
    )

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
        logger.info("Using layer-specialized prompts.")
    else:
        prompt_learner = RHEMCLIP_PromptLearner(model.to("cpu"), RHEMCLIP_parameters)
        prompt_learner.to(device)
        logger.info("Using shared prompts only.")

    # -------------------------
    # Learnable RAMF
    # -------------------------
    # reliability_estimator = LearnableReliabilityEstimator(
    #     in_channels=2,
    #     hidden_dim=args.reliability_hidden_dim
    # ).to(device)
    if args.use_reliability_fusion:
        reliability_estimator = LearnableReliabilityEstimator(
            in_channels=2,
            hidden_dim=args.reliability_hidden_dim
        ).to(device)
    else:
        reliability_estimator = None

    model.to(device)
  
    optim_params = list(prompt_learner.parameters())
    if args.use_reliability_fusion:
        optim_params += list(reliability_estimator.parameters())

    optimizer = torch.optim.Adam(
        optim_params,
        lr=args.learning_rate,
        betas=(0.5, 0.999)
    )
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    model.eval()
    prompt_learner.train()
    if args.use_reliability_fusion:
        reliability_estimator.train()

    for epoch in tqdm(range(args.epoch)):
        loss_list, image_loss_list = [], []

        for items in tqdm(train_dataloader):
            image = items['img'].to(device)
            label = items['anomaly'].to(device)
            gt = items['img_mask'].squeeze().to(device)
            gt = (gt > 0.5).float()

            with torch.no_grad():
        
                image_features, patch_features = model.encode_image(
                    image, args.features_list
                )
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        
            if args.use_layer_specialized_prompt:
                base_prompts, layer_prompts_list, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)

                # global image-level text feature
                text_features_global = model.encode_text_learn(
                    base_prompts, tokenized_prompts, compound_prompts_text
                ).float()
                text_features_global = torch.stack(torch.chunk(text_features_global, dim=0, chunks=2), dim=1)
                text_features_global = text_features_global / text_features_global.norm(dim=-1, keepdim=True)

                # layer-specific text features
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

         
            text_probs = image_features.unsqueeze(1) @ text_features_global.permute(0, 2, 1)
            text_probs = text_probs[:, 0, ...] / 0.07
            image_loss = F.cross_entropy(text_probs.squeeze(), label.long())
            image_loss_list.append(image_loss.item())

           
            layer_sim_maps = []
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

                    sim_map = RHEMCLIP_lib.get_similarity_map(
                        similarity[:, 1:, :], args.image_size
                    ).permute(0, 3, 1, 2)  # [B, 2, H, W]

                    layer_sim_maps.append(sim_map)
                    selected_layer_idx += 1

            stack_maps = torch.stack(layer_sim_maps, dim=1)  # [B, L, 2, H, W]

            if args.use_reliability_fusion:
                reliability_logits = []
                for i in range(stack_maps.shape[1]):
                    l_map = stack_maps[:, i, ...]
                    l_logit = reliability_estimator(l_map)
                    reliability_logits.append(l_logit)

                reliability_logits = torch.stack(reliability_logits, dim=1)
                layer_weights = F.softmax(reliability_logits, dim=1).unsqueeze(-1).unsqueeze(-1)
                final_anomaly_map = torch.sum(stack_maps * layer_weights, dim=1)
            else:
                final_anomaly_map = torch.mean(stack_maps, dim=1)
         
            loss_seg = 0.0

            if args.alpha > 0:
                for i in range(stack_maps.shape[1]):
                    l_map = stack_maps[:, i, ...]
                    loss_seg += args.alpha * (
                        loss_focal(l_map, gt) + loss_dice(l_map[:, 1, :, :], gt)
                    )

            if args.beta > 0:
                loss_seg += args.beta * (
                    loss_focal(final_anomaly_map, gt) +
                    loss_dice(final_anomaly_map[:, 1, :, :], gt)
                )

            loss_cons = torch.tensor(0.0, device=device)

            if args.consist_mode != 'none':
                img_prob = F.softmax(text_probs, dim=-1)[:, 1].detach()
                flat_map = final_anomaly_map[:, 1, :, :].reshape(final_anomaly_map.shape[0], -1)
                pixel_prob = torch.topk(
                    flat_map,
                    k=min(args.top_k, flat_map.shape[1]),
                    dim=1
                )[0].mean(dim=1)

                if args.consist_mode == 'symmetric':
                    loss_cons = F.mse_loss(pixel_prob, img_prob)
                elif args.consist_mode == 'asymmetric':
                    mask = (label == 1)
                    if mask.any():
                        loss_cons = F.relu(img_prob[mask] - pixel_prob[mask]).mean()

            total_loss = image_loss + args.lam * loss_seg + args.lambda_consistency * loss_cons

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_list.append(total_loss.item())

        if (epoch + 1) % args.print_freq == 0:
            logger.info(
                f'epoch [{epoch + 1}/{args.epoch}], '
                f'Total Loss: {np.mean(loss_list):.4f}, '
                f'Image Loss: {np.mean(image_loss_list):.4f}'
            )

        if (epoch + 1) % args.save_freq == 0:
            save_dict = {
                "prompt_learner": prompt_learner.state_dict(),
                "args": vars(args),
                "epoch": epoch + 1
            }

            if args.use_reliability_fusion:
                save_dict["reliability_estimator"] = reliability_estimator.state_dict()

            torch.save(
                save_dict,
                os.path.join(args.save_path, f'epoch_{epoch + 1}.pth')
            )
        if (epoch + 1) == args.lr_decay_epoch:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= args.lr_decay_gamma

            logger.info(
                f'Learning rate decayed to {optimizer.param_groups[0]["lr"]:.6f} '
                f'after epoch {epoch + 1}'
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser("RHEMCLIP")
    parser.add_argument("--train_data_path", type=str, default="./data/visa")
    parser.add_argument("--save_path", type=str, default='./checkpoint')
    parser.add_argument("--dataset", type=str, default='mvtec')

    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3])

    parser.add_argument("--epoch", type=int, default=15)
    # parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--lr_decay_epoch", type=int, default=1)  
    parser.add_argument("--lr_decay_gamma", type=float, default=0.5)  
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=518)

    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=0)

    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--print_freq", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=1)

    parser.add_argument("--lam", type=float, default=4)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--beta", type=float, default=1.0)

    parser.add_argument(
        "--consist_mode",
        type=str,
        default='none',
        choices=['none', 'symmetric', 'asymmetric']
    )
    parser.add_argument("--lambda_consistency", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--use_reliability_fusion", action="store_true")
    # layer-specialized prompt
    parser.add_argument("--use_layer_specialized_prompt", action="store_true")
    parser.add_argument("--layer_prompt_init_std", type=float, default=0.01)

    # learnable RAMF
    parser.add_argument("--reliability_hidden_dim", type=int, default=32)

    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
