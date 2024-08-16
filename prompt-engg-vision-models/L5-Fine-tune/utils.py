
import argparse
import copy
import gc
import logging
import math
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
import matplotlib.pyplot as plt
from matplotlib.image import imread
import comet_ml


check_min_version("0.28.0.dev0")

logger = get_logger(__name__)

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def save_model_card(
    repo_id: str,
    images=None,
    base_model=str,
    train_text_encoder=False,
    prompt=str,
    repo_folder=None,
    pipeline: DiffusionPipeline = None,
):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
# LoRA DreamBooth - {repo_id}

These are LoRA adaption weights for {base_model}. The weights were trained on {prompt} using [DreamBooth](https://dreambooth.github.io/). You can find some example images in the following. \n
{img_str}

LoRA for the text encoder was enabled: {train_text_encoder}.
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        prompt=prompt,
        model_description=model_description,
        inference=True,
    )
    tags = ["text-to-image", "diffusers", "lora", "diffusers-training"]
    if isinstance(pipeline, StableDiffusionPipeline):
        tags.extend(["stable-diffusion", "stable-diffusion-diffusers"])
    else:
        tags.extend(["if", "if-diffusers"])
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    pipeline,
    hyperparameters,
    accelerator,
    pipeline_args,
    epoch,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {hyperparameters.num_validation_images} images with prompt:"
        f" {hyperparameters.validation_prompt}."
    )
    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(hyperparameters.seed) if hyperparameters.seed else None

    if hyperparameters.validation_images is None:
        images = []
        for _ in range(hyperparameters.num_validation_images):
            with torch.cuda.amp.autocast():
                image = pipeline(**pipeline_args, generator=generator).images[0]
                images.append(image)
    else:
        images = []
        for image in hyperparameters.validation_images:
            image = Image.open(image)
            with torch.cuda.amp.autocast():
                image = pipeline(**pipeline_args, image=image, generator=generator).images[0]
            images.append(image)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {hyperparameters.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    torch.cuda.empty_cache()

    return images


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=512,
        center_crop=False,
        encoder_hidden_states=None,
        class_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = tokenize_prompt(
                self.tokenizer, self.instance_prompt, tokenizer_max_length=self.tokenizer_max_length
            )
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

            if self.class_prompt_encoder_hidden_states is not None:
                example["class_prompt_ids"] = self.class_prompt_encoder_hidden_states
            else:
                class_text_inputs = tokenize_prompt(
                    self.tokenizer, self.class_prompt, tokenizer_max_length=self.tokenizer_max_length
                )
                example["class_prompt_ids"] = class_text_inputs.input_ids
                example["class_attention_mask"] = class_text_inputs.attention_mask

        return example


def collate_fn(examples, with_prior_preservation=False):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
        if has_attention_mask:
            attention_mask += [example["class_attention_mask"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }

    if has_attention_mask:
        batch["attention_mask"] = attention_mask

    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


class DreamBoothTrainer:
    def __init__(self, hyperparameters):
        self.hyperparameters = dotdict(hyperparameters)

        for hyperparameter_name in self.default_hyperparameters:
            if hyperparameter_name not in self.hyperparameters:
                self.hyperparameters[hyperparameter_name] = self.default_hyperparameters[hyperparameter_name]

        set_seed(self.hyperparameters.seed)
        self.logging_dir = Path(self.hyperparameters.output_dir, self.hyperparameters.logging_dir)
        self.accelerator_project_config = ProjectConfiguration(
            project_dir=self.hyperparameters.output_dir,
            logging_dir=self.hyperparameters.logging_dir
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.hyperparameters.gradient_accumulation_steps,
            mixed_precision=self.hyperparameters.mixed_precision,
            log_with="comet_ml",
            project_config=self.accelerator_project_config,
        )


        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        if self.hyperparameters.output_dir is not None:
            os.makedirs(self.hyperparameters.output_dir, exist_ok=True)


    default_hyperparameters = {
        "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
        "revision": None,
        "variant": None,
        "tokenizer_name": None,
        "instance_data_dir": "./instance",
        "class_data_dir": './class',
        "instance_prompt": "A photo of a [V] man",
        "class_prompt": "A photo of a man",
        "with_prior_preservation": True,
        "prior_loss_weight": 1.0,
        "num_class_images": 100,
        "output_dir": "andrew-model",
        "seed": 4329,
        "resolution": 512,
        "center_crop": False,
        "train_batch_size": 4,
        "sample_batch_size": 4,
        "num_train_epochs": 10,
        "max_train_steps": None,
        "checkpointing_steps": 500,
        "checkpoints_total_limit": None,
        "resume_from_checkpoint": None,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True,
        "learning_rate": 5e-4,
        "scale_lr": False,
        "lr_scheduler": "constant",
        "lr_warmup_steps": 500,
        "lr_num_cycles": 1,
        "lr_power": 1.0,
        "dataloader_num_workers": 0,
        "use_8bit_adam": True if torch.cuda.is_available() else False,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_weight_decay": 1e-2,
        "adam_epsilon": 1e-08,
        "max_grad_norm": 1.0,
        "push_to_hub": False,
        "hub_token": None,
        "hub_model_id": None,
        "logging_dir": "logs",
        "allow_tf32": False,
        "report_to": "comet_ml",
        "mixed_precision": "fp16" if torch.cuda.is_available() else None,
        "prior_generation_precision": "fp16" if torch.cuda.is_available() else None,
        "local_rank": -1,
        "enable_xformers_memory_efficient_attention": True if torch.cuda.is_available() else False,
        "pre_compute_text_embeddings": True,
        "tokenizer_max_length": None,
        "text_encoder_use_attention_mask": False,
        "validation_images": None,
        "class_labels_conditioning": None,
        "rank": 4
    }


    def generate_class_images(self):
        class_images_dir = Path(self.hyperparameters.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < self.hyperparameters.num_class_images:
            torch_dtype = torch.float16 if self.accelerator.device.type == "cuda" else torch.float32
            if self.hyperparameters.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif self.hyperparameters.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif self.hyperparameters.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = DiffusionPipeline.from_pretrained(
                self.hyperparameters.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=self.hyperparameters.revision,
                variant=self.hyperparameters.variant,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = self.hyperparameters.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(self.hyperparameters.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=self.hyperparameters.sample_batch_size)

            sample_dataloader = self.accelerator.prepare(sample_dataloader)
            pipeline.to(self.accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not self.accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    
    def display_images(self, img_type="class"):
        if img_type == "class":
            folder_path = self.hyperparameters.class_data_dir
        else:
            folder_path = self.hyperparameters.instance_data_dir
        # List all files in the given folder
        files = os.listdir(folder_path)
        # Filter out the first six image files (assuming they are image files)
        images = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))][:6]

        plt.figure(figsize=(15, 10))
        for i, image_name in enumerate(images):
            # Load the image
            img = imread(os.path.join(folder_path, image_name))
            # Display the image
            plt.subplot(2, 3, i + 1) 
            plt.imshow(img)
            plt.title(image_name)
            plt.axis('off')
        plt.show()


    def initialize_models(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.hyperparameters.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=self.hyperparameters.revision,
            use_fast=False,
        )

        text_encoder_cls = import_model_class_from_model_name_or_path(self.hyperparameters.pretrained_model_name_or_path, self.hyperparameters.revision)
        text_encoder = text_encoder_cls.from_pretrained(
            self.hyperparameters.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.hyperparameters.revision, variant=self.hyperparameters.variant
        )
        vae = AutoencoderKL.from_pretrained(
            self.hyperparameters.pretrained_model_name_or_path, subfolder="vae", revision=self.hyperparameters.revision, variant=self.hyperparameters.variant
        )

        unet = UNet2DConditionModel.from_pretrained(
            self.hyperparameters.pretrained_model_name_or_path, subfolder="unet", revision=self.hyperparameters.revision, variant=self.hyperparameters.variant
        )

        # We only train the additional adapter LoRA layers
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)

        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move unet, vae and text_encoder to device and cast to weight_dtype
        unet.to(self.accelerator.device, dtype=weight_dtype)
        vae.to(self.accelerator.device, dtype=weight_dtype)
        text_encoder.to(self.accelerator.device, dtype=weight_dtype)

        if self.hyperparameters.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warning(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        if self.hyperparameters.gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        return tokenizer, text_encoder, vae, unet

    def unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def initialize_lora(self, unet):
        unet_lora_config = LoraConfig(
            r=self.hyperparameters.rank,
            lora_alpha=self.hyperparameters.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
        )
        unet.add_adapter(unet_lora_config)

        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                # there are only two options here. Either are just the unet attn processor layers
                # or there are the unet and text encoder atten layers
                unet_lora_layers_to_save = None
                text_encoder_lora_layers_to_save = None

                for model in models:
                    if isinstance(model, type(unwrap_model(unet))):
                        unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                    elif isinstance(model, type(unwrap_model(text_encoder))):
                        text_encoder_lora_layers_to_save = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(model)
                        )
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

                LoraLoaderMixin.save_lora_weights(
                    output_dir,
                    unet_lora_layers=unet_lora_layers_to_save,
                    text_encoder_lora_layers=text_encoder_lora_layers_to_save,
                )

        def load_model_hook(models, input_dir):
            unet_ = None
            text_encoder_ = None

            while len(models) > 0:
                model = models.pop()

                if isinstance(model, type(unwrap_model(unet))):
                    unet_ = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

            lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)

            unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
            unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
            incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")

            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

            # Make sure the trainable params are in float32. This is again needed since the base models
            # are in `weight_dtype`. More details:
            # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
            if self.hyperparameters.mixed_precision == "fp16":
                models = [unet_]
                #if hyperparameters.train_text_encoder:
                #  models.append(text_encoder_)

                # only upcast trainable parameters (LoRA) into fp32
                cast_training_params(models, dtype=torch.float32)

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)


        # Make sure the trainable params are in float32.
        if self.hyperparameters.mixed_precision == "fp16":
            models = [unet]
            #if hyperparameters.train_text_encoder:
            #   models.append(text_encoder)

            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models, dtype=torch.float32)

        return unet

    def initialize_optimizer(self, unet):
        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if self.hyperparameters.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # Optimizer creation
        params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))

        optimizer = optimizer_class(
            params_to_optimize,
            lr=self.hyperparameters.learning_rate,
            betas=(self.hyperparameters.adam_beta1, self.hyperparameters.adam_beta2),
            weight_decay=self.hyperparameters.adam_weight_decay,
            eps=self.hyperparameters.adam_epsilon,
        )

        return optimizer, params_to_optimize

    def prepare_dataset(self, tokenizer, text_encoder):

        def compute_text_embeddings(prompt):
            with torch.no_grad():
                text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=self.hyperparameters.tokenizer_max_length)
                prompt_embeds = encode_prompt(
                    text_encoder,
                    text_inputs.input_ids,
                    text_inputs.attention_mask,
                    text_encoder_use_attention_mask=self.hyperparameters.text_encoder_use_attention_mask,
                )

            return prompt_embeds

        pre_computed_encoder_hidden_states = compute_text_embeddings(self.hyperparameters.instance_prompt)
        validation_prompt_negative_prompt_embeds = compute_text_embeddings("")

        validation_prompt_encoder_hidden_states = None

        if self.hyperparameters.class_prompt is not None:
            pre_computed_class_prompt_encoder_hidden_states = compute_text_embeddings(self.hyperparameters.class_prompt)
        else:
            pre_computed_class_prompt_encoder_hidden_states = None

        text_encoder = None
        tokenizer = None

        gc.collect()
        torch.cuda.empty_cache()

        # Dataset and DataLoaders creation:
        train_dataset = DreamBoothDataset(
            instance_data_root=self.hyperparameters.instance_data_dir,
            instance_prompt=self.hyperparameters.instance_prompt,
            class_data_root=self.hyperparameters.class_data_dir if self.hyperparameters.with_prior_preservation else None,
            class_prompt=self.hyperparameters.class_prompt,
            class_num=self.hyperparameters.num_class_images,
            tokenizer=tokenizer,
            size=self.hyperparameters.resolution,
            center_crop=self.hyperparameters.center_crop,
            encoder_hidden_states=pre_computed_encoder_hidden_states,
            class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
            tokenizer_max_length=self.hyperparameters.tokenizer_max_length,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.hyperparameters.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples, self.hyperparameters.with_prior_preservation),
        )

        return train_dataset, train_dataloader


    def initialize_scheduler(self, train_dataloader, optimizer):
        # Scheduler and math around the number of training steps.
        self.hyperparameters.overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.hyperparameters.gradient_accumulation_steps)
        if self.hyperparameters.max_train_steps is None:
            self.hyperparameters.max_train_steps = self.hyperparameters.num_train_epochs * num_update_steps_per_epoch
            self.hyperparameters.overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            self.hyperparameters.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.hyperparameters.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.hyperparameters.max_train_steps * self.accelerator.num_processes,
            num_cycles=self.hyperparameters.lr_num_cycles,
            power=self.hyperparameters.lr_power,
        )

        return lr_scheduler

    def save_lora_weights(self, unet):
        # Save the lora layers
        unet = self.unwrap_model(unet)
        unet = unet.to(torch.float32)

        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        LoraLoaderMixin.save_lora_weights(
            save_directory=self.hyperparameters.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            text_encoder_lora_layers=None,
        )


def train(hyperparameters):
    experiment = comet_ml.Experiment()
    trainer = DreamBoothTrainer(hyperparameters)

    if trainer.hyperparameters.seed is not None:
        set_seed(trainer.hyperparameters.seed)

    trainer.generate_class_images()
    tokenizer, text_encoder, vae, unet = trainer.initialize_models()

    noise_scheduler = DDPMScheduler.from_pretrained(trainer.hyperparameters.pretrained_model_name_or_path, subfolder="scheduler")

    unet = trainer.initialize_lora(unet)
    optimizer, params_to_optimize = trainer.initialize_optimizer(unet)

    train_dataset, train_dataloader = trainer.prepare_dataset(tokenizer, text_encoder)
    lr_scheduler = trainer.initialize_scheduler(train_dataloader, optimizer)

    unet, optimizer, train_dataloader, lr_scheduler = trainer.accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Train!
    total_batch_size = trainer.hyperparameters.train_batch_size * trainer.hyperparameters.gradient_accumulation_steps

    global_step = 0
    epoch = 0

    progress_bar = tqdm(
        range(0, trainer.hyperparameters.max_train_steps),
        desc="Steps"
    )

    for epoch in range(0, trainer.hyperparameters.num_train_epochs):
        unet.train()

        for step, batch in enumerate(train_dataloader):
            with trainer.accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)

                # Convert images to latent space
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz, channels, height, width = model_input.shape

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                )
                
                timesteps = timesteps.long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = batch["input_ids"]

                if trainer.unwrap_model(unet).config.in_channels == channels * 2:
                    print("channels * 2 in_channels")
                    noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

                # Predict the noise residual
                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    encoder_hidden_states,
                    return_dict=False,
                )[0]

                # if model predicts variance, throw away the prediction. we will only train on the
                # simplified training objective. This means that all schedulers using the fine tuned
                # model must be configured to use one of the fixed variance variance types.
                if model_pred.shape[1] == 6:
                    print("Variance")
                    model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                target = noise

                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)

                # Compute instance loss
                instance_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                # Compute prior loss
                prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                # Add the prior loss to the instance loss.
                loss = instance_loss + trainer.hyperparameters.prior_loss_weight * prior_loss

                trainer.accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # Logs
            loss_metrics = {
                "loss": loss.detach().item(),
                "prior_loss": prior_loss.detach().item(),
                "instance_loss": instance_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }

            experiment.log_metrics(loss_metrics, step=global_step)

            progress_bar.set_postfix(**loss_metrics)
            progress_bar.update(1)



            if global_step >= trainer.hyperparameters.max_train_steps:
                break

        trainer.save_lora_weights(unet)
    experiment.add_tag(f"dreambooth-training")
    experiment.log_parameters(trainer.hyperparameters)
    trainer.accelerator.end_training()
