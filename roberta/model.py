from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed


def get_model(model_cfg):
    model = AutoModelForSequenceClassification.from_pretrained(model_cfg.model_name_or_path, return_dict=True)
    model = get_peft_model(model, model_cfg.peft_config)
    model.print_trainable_parameters()
    return model
