"""Count SAM feature parameters actually executed, including direct tensor use."""
import numpy as np
import torch


def sam_feature_parameter_accounting(predictor, image=None):
    model=predictor.model
    image=np.zeros((512,512,3),dtype=np.uint8) if image is None else image
    called=set();handles=[]
    try:
        for name,module in model.named_modules():
            handles.append(module.register_forward_hook(lambda module,args,out,name=name:called.add(name)))
        with torch.inference_mode():predictor.set_image(image)
    finally:
        for handle in handles:handle.remove()
    used=set()
    for name,module in model.named_modules():
        if name in called:used.update(id(param) for param in module.parameters(recurse=False))
    if getattr(predictor,'feature_source','decoder')=='decoder' and getattr(model,'directly_add_no_mem_embed',False):
        used.add(id(model.no_mem_embed))
    parameters={name:param.numel() for name,param in model.named_parameters() if id(param) in used}
    stored=sum(param.numel() for param in model.parameters())
    return dict(used_parameters=sum(parameters.values()),stored_parameters=stored,
        unused_parameters=stored-sum(parameters.values()),
        image_encoder_parameters=sum(value for name,value in parameters.items() if name.startswith('image_encoder.')),
        extra_feature_parameters={name:value for name,value in parameters.items() if not name.startswith('image_encoder.')},
        used_parameter_names=sorted(parameters),
        method='Unique parameters of executed modules plus directly added no_mem_embed;shared encoder counted once')
