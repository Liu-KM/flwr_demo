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
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # 加载模型的代码放在这里

model_cfg = {
    "model_name_or_path": "roberta-large",
    "peft_config": LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1),
}

def get_model(model_cfg):
    model = AutoModelForSequenceClassification.from_pretrained(model_cfg.model_name_or_path, return_dict=True)
    model = get_peft_model(model, model_cfg.peft_config)
    model.print_trainable_parameters()
    return model
