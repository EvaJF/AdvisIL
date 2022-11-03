from modified_shufflenet import ShuffleNetV2
from modified_resnet import ResNet, BasicBlock 
from modified_mobilenet import MobileNetV2
def model_namer(backbone, width_mult, depth_mult):
    model_name=backbone+'_w'+str(width_mult)+'_d'+str(depth_mult)
    return model_name


def modified_model_builder(num_classes, backbone, width_mult, depth_mult):
    assert backbone in ["resnetBasic", "resnetBottleneck", "mobilenet", "shufflenet"]
    if backbone == 'resnetBasic': # Resnet18 with basic blocks, layers = [2, 2, 2, 2] and width_per_group=64 constant
        model = ResNet(num_classes=num_classes, block=BasicBlock, layers = [2, 2, 2, 2], width_mult=width_mult, depth_mult=depth_mult)
    elif backbone == 'mobilenet' : #mobilenetv2
        model = MobileNetV2(num_classes=num_classes, width_mult=width_mult, depth_mult=depth_mult)
    elif backbone == 'shufflenet' : # shufflenetv2
        model = ShuffleNetV2(num_classes=num_classes, width_mult=width_mult, depth_mult=depth_mult)
    else : 
        print("Wrong backbone type provided")
    return model 
    
def count_params(model, trainable=True):
    """
    Computes the number of (trainable) parameters of a model.

    Args
        model : PyTorch
    """
    if trainable : 
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else : 
        pytorch_total_params = sum(p.numel() for p in model.parameters())
    return pytorch_total_params
