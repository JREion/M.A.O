# Independent code for MAO, do not rely on promptsrc.py
# trainer name: MAO_PromptSRC_Base

import copy
import os.path as osp
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .imagenet_templates import IMAGENET_TEMPLATES

from sentence_transformers import SentenceTransformer  # [MAO_Base] For semantic similarity measurement
from dassl.data.transforms.transforms import build_transform  # [MAO_Base] For image processing
from dassl.utils import read_image
import json
import random

_tokenizer = _Tokenizer()


# [DPC] In negative sampling stage, convert image data sampled from the image database
# DPC: Dual-Prompt Collaboration for Tuning Vision-Language Models (https://arxiv.org/abs/2503.13443)
def transform_image(cfg, img0, transform):
    def _transform_image(tfm, img0):
        img_list = []
        for k in range(1):
            img_list.append(tfm(img0))
        img = img_list
        if len(img) == 1:
            img = img[0]

        return img

    output = {}

    if isinstance(transform, (list, tuple)):
        for i, tfm in enumerate(transform):
            img = _transform_image(tfm, img0)
            keyname = "img"
            if (i + 1) > 1:
                keyname += str(i + 1)
            output[keyname] = img
    else:
        img = _transform_image(transform, img0)
        output["img"] = img  # [3, 224, 224]

    return output


# [DPC] Split the [absolute path] passed by DataLoader into suffix (relative path only) and prefix (the rest)
# Considered Windows (D:\\XXX\\dolphin/image_0011.jpg) and Linux (/usr/XXX/dolphin/image_0011.jpg)
def split_img_abs_path(abs_path, ref_path):
    split_sum = ref_path.count("/")  # count the number of "/" using path name as reference
    if "\\" in abs_path:
        split_result = abs_path.rsplit("\\", 1)  # Split based on the last "\\"
        path_prefix = split_result[0]
        path_suffix = split_result[1]
    elif "r'\'" in abs_path:
        split_result = abs_path.rsplit("r'\'", 1)  # Split based on the last "\"
        path_prefix = split_result[0]
        path_suffix = split_result[1]
    else:
        split_result = abs_path.rsplit("/", split_sum + 1)  # Split based on the n+1th "/" from the end
        path_prefix = split_result[0]
        path_suffix = split_result[1]
        if len(split_result) > 1:
            for split_id in range(2, len(split_result)):
                path_suffix = path_suffix + "/" + split_result[split_id]

    return path_prefix, path_suffix


# [DPC] Re-format the absolute path of ImageNet for DPC Dynamic Hard Negative Optimizer
def reformat_imagenet_path(path_str):
    # Windows or Linux path
    return re.sub(r'([\\/])train\1n\d{8}', '', path_str, count=1)


def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if not zero_shot_model:
        design_details = {"trainer": 'IVLP',
                          "vision_depth": cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_VISION,
                          "language_depth": cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT,
                          "vision_ctx": cfg.TRAINER.PROMPTSRC.N_CTX_VISION,
                          "language_ctx": cfg.TRAINER.PROMPTSRC.N_CTX_TEXT}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        # Return original CLIP model for generating frozen VL features
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
        return model
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch"
        n_ctx = cfg.TRAINER.PROMPTSRC.N_CTX_TEXT
        ctx_init = cfg.TRAINER.PROMPTSRC.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # [DPC] Extract learnable visual and textual vectors
        extracted_visual_ctx = clip_model.visual.transformer.ctx_list
        extracted_text_ctx = clip_model.transformer.ctx_list
        VPT_ctx = clip_model.visual.VPT

        if ctx_init and n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.PROMPTSRC.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        # Also create frozen CLIP
        clip_model_temp = load_clip_to_cpu(cfg, True).float().cuda()
        clip_model_temp_image = load_clip_to_cpu(cfg, True)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = clip_model_temp_image.visual
            # Now pre-compute the frozen VL embeddings
            all_teacher_features = []
            # Using multiple text templates to ensure textual diversity during training
            for single_template in IMAGENET_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                text_features = clip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))

        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1).mean(dim=1)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()
        # Compute the prompted image and text features
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))

        if self.prompt_learner.training:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            '''
            [MAO] Alterable Regularization: In the 'text_features' composed of all categories, according to the id of 
            'mao_label', extract the tensor with the index corresponding to the hard negative id of each sample in the 
            mini-batch separately to form a new 'text_features_mao' with a size of [bs*TopK, 512]
            '''
            text_features_mao = text_features[label.tolist()]
            text_features_mao = text_features_mao / text_features_mao.norm(dim=-1, keepdim=True)
            # PromptSRC backbone
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # Compute the prompted logits
            logits = logit_scale * image_features @ text_features.t()

            logits_mao = logit_scale * image_features @ text_features_mao.t()
            label_mao_ids = torch.arange(label.size(0), device=logits.device).long()

            # Now calculate the frozen pre-trained features
            fixed_embeddings = self.prompt_learner.fixed_embeddings  # precomputed pre-trained frozen textual features
            fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                zero_shot_features = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
                zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
                # Compute pre-trained frozen visual features
                zero_shot_logits = logit_scale * zero_shot_features.cuda() @ fixed_embeddings.half().cuda().t()

            return F.cross_entropy(logits_mao,
                                   label_mao_ids), text_features, fixed_embeddings, zero_shot_features, \
                   image_features, zero_shot_logits, logits
        else:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # Compute the prompted logits
            logits = logit_scale * image_features @ text_features.t()
            return logits


@TRAINER_REGISTRY.register()
class MAO_PromptSRC_Base(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.transform_img = build_transform(cfg, is_train=True)  # [DPC] Introduce
        pic_lib = cfg.MAO.PIC_LIB  # [MAO_Base] Path of image database

        # [MAO_Base] Load pic_lib
        with open(pic_lib) as f:
            '''
            The format of 'pics_for_selection' dict is like:
            {
                'train': [{'face': [0, ['1.jpg', '2.jpg']], 'leopard': [1, ['3.jpg', '4.jpg']], ... }],
                'val': [{'face': [0, ['5.jpg', '6.jpg']], 'leopard': [1, ['7.jpg', '8.jpg']], ... }],
                'train_obj_list': ['face', 'leopard', ...],
                'val_obj_list': ['face', 'leopard', ...]
            }
            - This dict can be found in './DATA/SPLE_database' folder.
            - The list length of the 'train' and 'val' values is always 1.
            - DO NOT load val when fine-tuning to avoid data leakage.
            '''
            self.pics_for_selection = json.load(f)

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTSRC.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PROMPTSRC.PREC == "fp32" or cfg.TRAINER.PROMPTSRC.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)
        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        N = cfg.OPTIM.MAX_EPOCH
        mean = cfg.TRAINER.PROMPTSRC.GPA_MEAN
        stdev = cfg.TRAINER.PROMPTSRC.GPA_STD
        gauss = self.get_gauss(mean, stdev)
        self.gauss = np.array([gauss(a) for a in range(1, N + 1)])
        self.gauss = self.gauss / sum(self.gauss)
        self.scaler = GradScaler() if cfg.TRAINER.PROMPTSRC.PREC == "amp" else None
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
        # Keep model with GPA
        self.previous_model_gpa = None

        # [MAO_Base] Introduce MiniLM-L6-v2 semantic measurement model
        print("[MAO_Base] Import Semantic Measurement Model for Data-Driven Enhancement")
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        class_embeds = self.semantic_model.encode(classnames)  # Encode all BASE categories
        self.class_embeds_torch = torch.tensor(class_embeds)  # Convert to PyTorch tensor

    def forward_backward(self, batch):
        """
        [MAO_Base] Data-Driven Enhancement:
        For a mini-batch of length [bs], use the MAO method to obtain hard negative image-text pairs,
        making the length of mini-batch to [bs*TopK]
        """
        image, label = self.parse_mao_batch_train_norepeat(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.PROMPTSRC.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss_ce, normalized_text_features, zs_clip_text_embeddings, zs_image_embedd, image_ft, \
            zero_shot_logits, logits = model(image, label)
            # Calculate the L_SCL_text loss
            loss_scl_text = F.l1_loss(normalized_text_features, zs_clip_text_embeddings.cuda(),
                                      reduction='mean') * self.cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT
            # Calculate the L_SCL_image loss
            loss_scl_image = F.l1_loss(image_ft, zs_image_embedd.cuda(),
                                       reduction='mean') * self.cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT
            # Now calculate L_SCL_logits
            L_SCL_logits = F.kl_div(
                F.log_softmax(logits / 1, dim=1),
                F.log_softmax(zero_shot_logits / 1, dim=1),
                reduction='sum',
                log_target=True
            ) * (1 * 1) / logits.numel()
            L_SCL = (L_SCL_logits + loss_scl_text + loss_scl_image)
            loss = (loss_ce + L_SCL)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            # Means one epoch is completed, perform GPA
            self.step_counter = self.step_counter + 1
            current_epoch_weight = self.gauss[self.step_counter - 2]
            current_model_weights = copy.deepcopy(model.state_dict())
            weighted_state_dict = self.state_dict_weighting(current_model_weights, current_epoch_weight)
            if self.previous_model_gpa is None:
                self.previous_model_gpa = weighted_state_dict
            else:
                self.previous_model_gpa = self.state_dict_add(weighted_state_dict, self.previous_model_gpa)

        if self.step_counter == self.model.total_epochs + 1:
            print("Using GPA model for final inference...")
            model.load_state_dict(self.previous_model_gpa)
            self.model.load_state_dict(self.previous_model_gpa)
        return loss_summary

    def state_dict_weighting(self, main_dict, weightage, prompt_only=False):
        # Average all parameters
        updated_dict = copy.deepcopy(main_dict)
        if not prompt_only:
            for key in main_dict:
                updated_dict[key] = main_dict[key] * weightage
            return updated_dict
        else:
            return main_dict * weightage

    def state_dict_add(self, dict1, dict2, prompt_only=False):
        # Average all parameters
        if not prompt_only:
            modified_dict = dict2
            for key in dict1:
                modified_dict[key] = (modified_dict[key] + dict1[key])
            return modified_dict
        else:
            return dict1 + dict2

    def get_gauss(self, mu, sigma):
        gauss = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return gauss

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def parse_mao_batch_train_norepeat(self, batch):
        """
        [MAO_Base] Data-Driven Enhancement:
        - For a given mini-batch, measure the semantic similarity of a given label in the base class and take the
          Top-K as hard negative objects.
        - Then, randomly extract images based on the hard negative object to form a new mini-batch containing
          ground-truth and Top-K hard negative objects.
        - Note: Since only the labels in the mini-batch are used as candidates when calculating the cross entropy loss,
          the [norepeat] setting also needs to be followed.
        """
        input = batch["img"]  # torch.Size([bs,3,224,224])
        label = batch["label"]  # torch.Size([bs])
        img_path = batch["impath"]  # torch.Size([bs])

        # Read and extend config
        cfg = self.cfg
        class_label = self.dm.dataset.classnames
        topk_sum = cfg.MAO.INFER_TOPK  # The number (Top-K) of sampled hard negatives

        with torch.no_grad():
            # Init
            input_mao = torch.empty(0, 3, 224, 224)
            label_mao = torch.empty(0)
            objects_in_batch = label.tolist()

            # Take the original passed labels of classname
            input_name = [class_label[i] for i in label]
            re_input_name = [name.replace("_", " ") for name in input_name]
            input_embedding = self.semantic_model.encode(re_input_name)
            input_embedding_torch = torch.tensor(input_embedding)
            class_embeddings_torch = self.class_embeds_torch  # Embeddings of all candidate classes
            class_norm = class_embeddings_torch / class_embeddings_torch.norm(dim=1, keepdim=True)

            # For each input image and label in the batch, perform Top-K reasoning
            for sample_id in range(0, input.size(0)):
                input_embedding_sample = input_embedding_torch[sample_id]
                current_input_norm = input_embedding_sample / input_embedding_sample.norm(dim=0)  # norm

                # Calculate cosine similarity
                similarities = torch.mm(class_norm, current_input_norm.unsqueeze(1)).squeeze(1)
                indice = torch.topk(similarities, k=topk_sum).indices  # Extract Top-K index from similarity

                # Object Filtering: sample Top K-1 negative samples other than ground-truth as hard negative objects
                hn_labels_before_selection = []
                hn_labels = []
                for index in indice:
                    if index != label[sample_id] and len(hn_labels) < topk_sum - 1:
                        hn_labels_before_selection.append(index)

                # non-repeat filtering
                for item in hn_labels_before_selection:
                    if item not in objects_in_batch:
                        hn_labels.append(item)
                        objects_in_batch.append(item)

                # If 'hn_labels' is empty, then randomly select 2 base-class objects outside from 'objects_in_batch'
                if len(hn_labels) < 2:
                    for step in range(0, len(class_label) - 1):
                        neg_label = random.randint(1, len(class_label) - 1)
                        if neg_label not in objects_in_batch and len(hn_labels) < 2:
                            hn_labels.append(neg_label)
                            objects_in_batch.append(neg_label)
                        elif len(hn_labels) < 2:
                            continue
                        else:
                            break

                # Image Sampling: use 'hn_labels' as query to sample positive images in training set
                hn_pic_paths = []
                for i in range(0, len(hn_labels)):
                    hn_obj_name = class_label[hn_labels[i]]  # Get classname
                    pic_list = self.pics_for_selection["train"][0].get(hn_obj_name)
                    random_pic_path = random.choice(pic_list[1])
                    hn_pic_paths.append(random_pic_path)

                # Read the image and convert it to the CLIP standard input format with a size of [3,224,224]
                input_for_concat = input[sample_id].unsqueeze(0).to(self.device)
                dataset_name = cfg.DATASET.NAME
                # special path format (EuroSAT and ImageNet)
                if dataset_name == "EuroSAT":
                    img_path_prefix, _ = split_img_abs_path(img_path[sample_id], "Highway/Highway_2417.jpg")
                elif dataset_name == "ImageNet":
                    img_path_prefix_cache, _ = split_img_abs_path(img_path[sample_id], "n1234567_1.JPEG")
                    img_path_prefix = reformat_imagenet_path(img_path_prefix_cache)
                else:
                    img_path_prefix, _ = split_img_abs_path(img_path[sample_id], random_pic_path)

                for processing_img in hn_pic_paths:
                    img0 = read_image(img_path_prefix + "/" + processing_img)  # Use ABSOLUTE PATH to read image
                    transformed_img = transform_image(cfg, img0, self.transform_img)["img"].to(
                        self.device)  # Transform image
                    input_for_concat = torch.cat([input_for_concat,
                                                  transformed_img.unsqueeze(0)
                                                  ],
                                                 dim=0)

                label_for_concat = torch.cat([label[sample_id].unsqueeze(0),
                                              torch.Tensor(hn_labels)],
                                             dim=0)

                label_mao = torch.cat([label_mao, label_for_concat], dim=0)
                input_mao = torch.cat([input_mao.to(self.device), input_for_concat], dim=0)

            # Build final mini-batch
            input_mao = input_mao.to(self.device)
            label_mao = label_mao.type(label.dtype).to(self.device)

        return input_mao, label_mao

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            if epoch < 0:
                all_model_files = os.listdir(osp.join(directory, name))
                all_model_files = [file_ for file_ in all_model_files if file_ != 'checkpoint']
                model_epochs = [int(file_.split('-')[-1]) for file_ in all_model_files]
                last_epoch = max(model_epochs)
                model_file = 'model.pth.tar-' + str(last_epoch)

            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            for del_name, param in self.model.named_parameters():
                if not param.requires_grad:
                    del state_dict[del_name]
            print("====== [DPC] filtered_state_dict ======")
            print("length of filtered state_dict: ", len(state_dict))
            key_list = []
            for key in state_dict:
                key_list.append(key)
            print(key_list)

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)