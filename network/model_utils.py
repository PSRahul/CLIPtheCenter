def set_parameter_requires_grad(model, freeze_params):
    for param in model.parameters():
        param.requires_grad = not freeze_params
    return model