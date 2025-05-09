# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from utils.gen_utils import AverageMeter
from utils.metric_utils import get_recall

from tqdm import tqdm

@torch.no_grad()
def validate_global(clip_model, combiner, val_image_loader, valloader_global, topk=(1, 5, 10), save_path=None):
    
    combiner.eval()
    clip_model.eval()

    print('Evaluating in GLOBAL setting, extacting features for all images...')
    gallery_ranks = []
    gallery_feats = []
    for batch in tqdm(val_image_loader):

        image, rank = batch
        image = image.cuda()

        image_feature = clip_model.encode_image(image).float()
        image_feature = torch.nn.functional.normalize(image_feature, p=2, dim=-1)

        gallery_feats.append(image_feature)
        gallery_ranks.append(rank.cuda())
        
    gallery_ranks = torch.cat(gallery_ranks)
    gallery_feats = torch.cat(gallery_feats)

    print('Performing eval using Combiner...')
    meters = {k: AverageMeter() for k in topk}
    sims_to_save = []

    with torch.no_grad():
        for batch in tqdm(valloader_global):

            ref_img, caption, ref_global_rank, target_global_rank = [x.cuda(non_blocking=True) for x in batch]
            caption = caption.squeeze()
            if len(target_global_rank.size()) == 1:
                target_global_rank = target_global_rank.unsqueeze(-1) 

            # Forward pass in CLIP
            ref_feats = clip_model.encode_image(ref_img).float()
            caption_feats = clip_model.encode_text(caption).float()

            # Forward pass in combiner
            combined_feats = combiner(ref_feats, caption_feats)
            combined_feats = torch.nn.functional.normalize(combined_feats, p=2, dim=-1)

            # Compute similarity
            similarities = combined_feats[:, None, :] * gallery_feats       # B x N x D
            similarities = similarities.sum(dim=-1)                         # B x N

            # Now mask some similarities (set to -inf) if they correspond the to the same feature in the gallery
            mask = ref_global_rank[:, None].cuda() == gallery_ranks
            similarities[mask] = -1e5

            # Sort the similarities in ascending order (closest example is the predicted sample)
            _, sort_idxs = similarities.sort(dim=-1, descending=True)                   # B x N

            # Compute recall at K
            for k in topk:

                recall_k = get_recall(sort_idxs[:, :k], target_global_rank)
                meters[k].update(recall_k, len(ref_img))

            sims_to_save.append(similarities.cpu())

        # Print results
        print_str = '\n'.join([f'Recall @ {k} = {v.avg:.4f}' for k, v in meters.items()])
        print(print_str)

        if save_path is not None:
            sims_to_save = torch.cat(sims_to_save)
            print(f'Saving text only preds to: {save_path}')
            torch.save(sims_to_save, save_path)
        
        return meters

@torch.no_grad()
def validate(clip_model, combiner, valloader, topk=(1, 2, 3), save_path=None, models=None, img_weight=0):
    import torch.nn.functional as F
    print('Computing eval with combiner...')

    # clip_model.eval()
    # combiner.eval()

    meters = {k: AverageMeter() for k in topk}
    sims_to_save = []

    print(f'Computing eval WITHOUT trained combiner... img_weight: {img_weight}')

    for key in models.keys():
        models[key].float().eval()
    text_encoder = models['text_encoder']
    image_encoder = models['image_encoder']

    def combiner(ref_feats, caption_feats):
        composed_embeds = img_weight * F.normalize(ref_feats, dim=-1) + F.normalize(caption_feats, dim=-1)
        return F.normalize(composed_embeds, dim=-1)

    with torch.no_grad():
        for batch in tqdm(valloader):

            ref_img, caption, gallery_set, target_rank = [x.cuda(non_blocking=True) for x in batch[:4]]
            bsz, n_gallery, _, h, w = gallery_set.size()
            caption = caption.squeeze()

            # Forward pass in CLIP
            # imgs_ = torch.cat([ref_img, gallery_set.view(-1, 3, h, w)], dim=0)
            # all_img_feats = clip_model.encode_image(imgs_).float()
            # caption_feats = clip_model.encode_text(caption).float()
            imgs_ = torch.cat([ref_img, gallery_set.view(-1, 3, h, w)], dim=0)
            all_img_feats = image_encoder(imgs_).float()
            caption_feats = text_encoder(caption).float()

            # L2 normalize and view into correct shapes
            ref_feats, gallery_feats = all_img_feats.split((bsz, bsz * n_gallery), dim=0)
            gallery_feats = gallery_feats.view(bsz, n_gallery, -1)
            gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=-1)

            # Forward pass in combiner
            combined_feats = combiner(ref_feats, caption_feats)

            # Compute similarity
            similarities = combined_feats[:, None, :] * gallery_feats       # B x N x D
            similarities = similarities.sum(dim=-1)                         # B x N

            # Sort the similarities in ascending order (closest example is the predicted sample)
            _, sort_idxs = similarities.sort(dim=-1, descending=True)                   # B x N

            # Compute recall at K
            for k in topk:

                recall_k = get_recall(sort_idxs[:, :k], target_rank)
                meters[k].update(recall_k, bsz)

            sims_to_save.append(similarities.cpu())

        if save_path is not None:
            sims_to_save = torch.cat(sims_to_save)
            print(f'Saving text only preds to: {save_path}')
            torch.save(sims_to_save, save_path)

        # Print results
        print_str = '\n'.join([f'Recall @ {k} = {v.avg:.4f}' for k, v in meters.items()])
        print(print_str)

        return meters