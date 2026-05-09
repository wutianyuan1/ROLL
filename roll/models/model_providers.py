import os
from typing import Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizer,
    TrainingArguments,
)
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled


try:
    from mcore_adapter import TrainingArguments as mca_TrainingArguments
    from mcore_adapter.models import AutoModel
except Exception as e:
    mca_TrainingArguments = None

from roll.configs import ModelArguments
from roll.utils.checkpoint_manager import download_model
from roll.utils.logging import get_logger


logger = get_logger()


def default_tokenizer_provider(model_args: "ModelArguments"):
    model_name_or_path = download_model(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        split_special_tokens=False,
        trust_remote_code=True,
        padding_side="left",
    )
    return tokenizer


def default_processor_provider(model_args: "ModelArguments"):
    model_name_or_path = download_model(model_args.model_name_or_path)
    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    except Exception as e:
        logger.info(f"processor not found: {e}")
        processor = None
    return processor


def load_valuehead_params(model_path):
    """
    modified from llamafactory
    """
    err_text = ""

    try:
        from safetensors import safe_open

        vhead_file = os.path.join(model_path, "value_head.safetensors")
        with safe_open(vhead_file, framework="pt", device="cpu") as f:
            return {key: f.get_tensor(key) for key in f.keys()}
    except Exception as err:
        err_text = str(err)

    try:
        vhead_file = os.path.join(model_path, "value_head.bin")
        return torch.load(vhead_file, map_location="cpu")
    except Exception as err:
        err_text = str(err)

    logger.info("Provided path ({}) does not contain value head weights: {}.".format(model_path, err_text))
    logger.info("Ignore the above message if you are not resuming the training of a value head model.")
    return None


def freeze_model(model, model_args: "ModelArguments"):
    if model_args.freeze_module_prefix is None:
        return

    prefixes = model_args.freeze_module_prefix.split(",")
    logger.info(f"Freeze model with prefix: {prefixes}")
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            param.requires_grad_(False)


def load_model(
    model_args: "ModelArguments",
    is_trainable: Optional[bool] = False,
    add_valuehead: Optional[bool] = False,
):
    r"""
    modified from llamafactory
    """
    model_name_or_path = download_model(model_args.model_name_or_path)
    init_kwargs = {"trust_remote_code": True, **model_args.model_config_kwargs}
    config = AutoConfig.from_pretrained(model_name_or_path, **init_kwargs)
    if model_args.attn_implementation is not None and model_args.attn_implementation != "auto":
        setattr(config, "attn_implementation", model_args.attn_implementation)
    if not is_trainable:
        setattr(config, "use_cache", True)
    if model_args.moe_aux_loss_coef is not None:
        setattr(config, "router_aux_loss_coef", model_args.moe_aux_loss_coef)
        setattr(config, "output_router_logits", is_trainable)
    init_kwargs["low_cpu_mem_usage"] = not is_deepspeed_zero3_enabled()
    if not is_deepspeed_zero3_enabled() and not is_fsdp_enabled():
        init_kwargs["torch_dtype"] = model_args.compute_dtype
        if init_kwargs["low_cpu_mem_usage"]:  # device map requires low_cpu_mem_usage=True
            if "device_map" not in init_kwargs and model_args.device_map:
                init_kwargs["device_map"] = model_args.device_map

    init_kwargs["config"] = config
    init_kwargs["pretrained_model_name_or_path"] = model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(**init_kwargs)
    if not model_args.disable_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    freeze_model(model, model_args)

    if add_valuehead:
        from trl import AutoModelForCausalLMWithValueHead

        model = AutoModelForCausalLMWithValueHead.from_pretrained(model, **model_args.model_config_kwargs)

        vhead_params = load_valuehead_params(model_name_or_path)
        if vhead_params is not None:
            if is_deepspeed_zero3_enabled():
                import deepspeed  # type: ignore

                params = [param for _, param in model.v_head.named_parameters(recurse=False)]
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        model.load_state_dict(vhead_params, strict=False)
            else:
                model.load_state_dict(vhead_params, strict=False)
            logger.info("Loaded valuehead from checkpoint: {}".format(model_name_or_path))

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    return model


def default_actor_model_provider(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    training_args: "TrainingArguments" = None,
    is_trainable: Optional[bool] = False,
):
    old_model_name_or_path = model_args.model_name_or_path
    model_args.model_name_or_path = download_model(model_args.model_name_or_path)
    if (
        mca_TrainingArguments is not None
        and training_args is not None
        and isinstance(training_args, mca_TrainingArguments)
    ):
        # megatron
        if model_args.moe_aux_loss_coef is not None and training_args.moe_aux_loss_coeff is None:
            training_args.moe_aux_loss_coeff = model_args.moe_aux_loss_coef
        model = AutoModel.from_pretrained(model_args.model_name_or_path, training_args)
        freeze_model(model, model_args)
        if is_trainable:
            model.train()
            for param in model.parameters():
                param.requires_grad = True
        else:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
    else:
        # hf
        init_kwargs = {
            "torch_dtype": model_args.compute_dtype,
            "trust_remote_code": True,
        }
        if not is_deepspeed_zero3_enabled():
            init_kwargs["low_cpu_mem_usage"] = True
            if is_trainable:
                init_kwargs["device_map"] = {"": torch.cuda.current_device()}
            elif model_args.device_map:
                init_kwargs["device_map"] = model_args.device_map
            elif model_args.export_dir is None:
                init_kwargs["device_map"] = "balanced"
        logger.info(f"init_kwargs: {init_kwargs}")
        model = load_model(model_args, is_trainable, False)
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

    model_args.model_name_or_path = old_model_name_or_path
    return model


def default_reward_model_provider(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    training_args: "TrainingArguments" = None,
    is_trainable: Optional[bool] = False,
):
    """
    model.forward 遵循TokenClassifierOutput 协议
    class TokenClassifierOutput(ModelOutput):
        logits: torch.FloatTensor   # 必须要有
        loss: Optional[torch.FloatTensor] = None
        hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
        attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    """
    old_model_name_or_path = model_args.model_name_or_path
    model_args.model_name_or_path = download_model(model_args.model_name_or_path)

    if (
        mca_TrainingArguments is not None
        and training_args is not None
        and isinstance(training_args, mca_TrainingArguments)
    ):
        # megatron
        raise NotImplementedError("megatron reward model not implemented")
    else:
        init_kwargs = {
            "torch_dtype": model_args.compute_dtype,
            "trust_remote_code": True,
        }
        if not is_deepspeed_zero3_enabled():
            init_kwargs["low_cpu_mem_usage"] = True
            if is_trainable:
                init_kwargs["device_map"] = {"": torch.cuda.current_device()}
            elif model_args.device_map:
                init_kwargs["device_map"] = model_args.device_map
            elif model_args.export_dir is None:
                init_kwargs["device_map"] = "auto"
        logger.info(f"init_kwargs: {init_kwargs}")
        if model_args.model_type in ["auto_sequence_classification"]:
            logger.info(f"use AutoModelForSequenceClassification model {model_args.model_type}")
            attn_implementation = "flash_attention_2" if model_args.flash_attn == "fa2" else "eager"
            config = AutoConfig.from_pretrained(model_args.model_name_or_path)
            config.num_labels = model_args.num_labels
            setattr(config, "attn_implementation", attn_implementation)
            setattr(config, "_attn_implementation", attn_implementation)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path, config=config, **init_kwargs
            )
        elif model_args.model_type in ["auto_token_classification"]:
            logger.info(f"use AutoModelForTokenClassification model {model_args.model_type}")
            attn_implementation = "flash_attention_2" if model_args.flash_attn == "fa2" else "eager"
            config = AutoConfig.from_pretrained(model_args.model_name_or_path)
            config.num_labels = model_args.num_labels
            setattr(config, "attn_implementation", attn_implementation)
            setattr(config, "_attn_implementation", attn_implementation)
            model = AutoModelForTokenClassification.from_pretrained(
                model_args.model_name_or_path, config=config, **init_kwargs
            )
        elif model_args.model_type in ["trl"]:
            from trl import AutoModelForCausalLMWithValueHead

            from roll.models.trl_patches import (
                no_set_device_hook_post_init,
                token_classifier_forward,
                value_head_load_state_dict,
            )

            AutoModelForCausalLMWithValueHead.post_init = no_set_device_hook_post_init
            model = load_model(model_args, is_trainable, True)
            setattr(model, "forward", token_classifier_forward.__get__(model))
            setattr(model, "load_state_dict", value_head_load_state_dict.__get__(model))
            logger.info(f"patch AutoModelForCausalLMWithValueHead load_state_dict and forward")
        else:
            raise NotImplementedError
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

    model_args.model_name_or_path = old_model_name_or_path

    return model


def default_value_model_provider(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    training_args: "TrainingArguments" = None,
    is_trainable: Optional[bool] = False,
):
    """
    TokenClassifierOutput
    """
    old_model_name_or_path = model_args.model_name_or_path
    model_args.model_name_or_path = download_model(model_args.model_name_or_path)

    if (
        mca_TrainingArguments is not None
        and training_args is not None
        and isinstance(training_args, mca_TrainingArguments)
    ):
        raise NotImplementedError("megatron value model not implemented")
    else:
        init_kwargs = {
            "torch_dtype": model_args.compute_dtype,
            "trust_remote_code": True,
        }
        if not is_deepspeed_zero3_enabled():
            init_kwargs["low_cpu_mem_usage"] = True
            if is_trainable:
                init_kwargs["device_map"] = {"": torch.cuda.current_device()}
            elif model_args.device_map:
                init_kwargs["device_map"] = model_args.device_map
            elif model_args.export_dir is None:
                init_kwargs["device_map"] = "auto"
        logger.info(f"init_kwargs: {init_kwargs}")
        if model_args.model_type in ["auto_token_classification"]:
            logger.info(f"use AutoModelForTokenClassification model {model_args.model_type}")
            attn_implementation = "flash_attention_2" if model_args.flash_attn == "fa2" else "eager"
            config = AutoConfig.from_pretrained(model_args.model_name_or_path)
            config.num_labels = model_args.num_labels
            setattr(config, "attn_implementation", attn_implementation)
            setattr(config, "_attn_implementation", attn_implementation)
            model = AutoModelForTokenClassification.from_pretrained(
                model_args.model_name_or_path, config=config, **init_kwargs
            )
        elif model_args.model_type in ["trl"]:
            from trl import AutoModelForCausalLMWithValueHead

            from roll.models.trl_patches import (
                no_set_device_hook_post_init,
                token_classifier_forward,
                value_head_load_state_dict,
            )

            AutoModelForCausalLMWithValueHead.post_init = no_set_device_hook_post_init
            model = load_model(model_args, is_trainable, True)
            setattr(model, "forward", token_classifier_forward.__get__(model))
            setattr(model, "load_state_dict", value_head_load_state_dict.__get__(model))
        else:
            raise NotImplementedError
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

    model_args.model_name_or_path = old_model_name_or_path

    return model
