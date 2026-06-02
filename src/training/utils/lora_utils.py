import torch
from peft import LoraConfig, get_peft_model, TaskType

def get_finetuning_model(net, use_lora=False, lora_r=8, lora_alpha=32, lora_dropout=0.1):
    freeze_names = [
        "camera_head.",
        "point_head.",
        'depth_head.',
        "aggregator.camera_token",
        "aggregator.register_token",
        "aggregator.patch_embed.",
    ]
    
    lora_names = [
        "aggregator.frame_blocks.",
        "aggregator.global_blocks.",
    ]
    
    train_names = [
        "aggregator.rgb_token",
        "aggregator.patch_embed_ray",
        "aggregator.patch_embed_proj",
    ]
    
    if use_lora:
        trainable_state = {}
        for name, module in net.named_parameters():
            trainable_state[name] = module.requires_grad
        
        # print(len(trainable_state), list(trainable_state.keys())[:10])
        
        lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=find_all_linear_names(net, lora_names),  # depends on model architecture
                lora_dropout=lora_dropout,
                bias="none",
            )
        net = get_peft_model(net, lora_config)
    
    for name, module in net.named_parameters():
        # print(name)
        if any(freeze_name in name for freeze_name in freeze_names):
            module.requires_grad = False
        # elif any(train_name in name for train_name in train_names):
        #     # module.requires_grad = True
        #     pass
        # elif any(lora_name in name for lora_name in lora_names) and (not use_lora):
        #     # module.requires_grad = True
        #     pass
        elif use_lora and (not any(lora_name in name for lora_name in lora_names)):
            # if use lora & not lora layer, use the default trainable state
            if name[len('base_model.model.'):] in trainable_state:
                module.requires_grad = trainable_state[ name[len('base_model.model.'):] ]
            else:
                print('module', name, 'requires_grad', module.requires_grad)
        else:
            # print('module', name, 'requires_grad', module.requires_grad)
            pass
            
    return net


def find_all_linear_names(model, lora_names):
    cls = torch.nn.Linear
    lora_module_names = set()
    lora_target_modules = ['qkv']
   
    for name, module in model.named_modules():
        if isinstance(module, cls) and \
            any(lora_name in name for lora_name in lora_names) and \
                any(lora_target_module in name for lora_target_module in lora_target_modules):
            lora_module_names.add(name)

    print('lora module', lora_module_names)
    return list(lora_module_names)


def get_peft_state(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError

    return to_return


def save_trainable_parameters(model, filename="logs/trainable_parameters.txt"):
    """
    Saves the names, shapes, and number of trainable parameters of a PyTorch model to a text file.
    """
    total_trainable_params = 0
    with open(filename, 'w') as f:
        f.write("Trainable Parameters Report\n")
        f.write("="*50 + "\n")
        for name, param in model.named_parameters():
            saved_name  = name if "base_model.model." not in name else name.split("base_model.model.")[1]
            if param.requires_grad:
                num_params = param.numel()
                total_trainable_params += num_params
                f.write(f"{saved_name:<60} | Shape: {str(param.shape):<25} | Params: {num_params}\n")

            else:
                f.write(f"{saved_name:<60} | Shape: {str(param.shape):<25} | Requires Grad: {param.requires_grad}\n")
                
        f.write("="*50 + "\n")
        f.write(f"Total Trainable Parameters: {total_trainable_params}\n")
        
    print(f"✅ Trainable parameters saved to {filename}")
    print(f"Total Trainable Parameters: {total_trainable_params}")
