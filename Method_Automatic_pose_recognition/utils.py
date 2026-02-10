def checkDirMake(directory):
    #print(directory)
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False