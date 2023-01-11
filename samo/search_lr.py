from torch_lr_finder import LRFinder, TrainDataLoaderIter
from main import *


# See also: https://github.com/davidtvs/pytorch-lr-finder/blob/acc5e7e/torch_lr_finder/lr_finder.py#L31-L41
class MyTrainDataLoaderIter(TrainDataLoaderIter):
    def inputs_labels_from_batch(self, batch_data):
        _, labels, _, _, _ = batch_data
        inputs = batch_data
        return inputs, labels  # match (inputs, label)


# NOTE: Use this wrapper to unpack input data
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, inputs):
        feat, labels, spk, utt, tag = inputs  # unpack
        feats, fc_outputs = self.model(feat)
        return (feats, spk)


# NOTE: Use this wrapper to unpack outputs
class LossFunctionWrapper(nn.Module):
    def __init__(self, loss_func, centers):
        super(LossFunctionWrapper, self).__init__()
        self.loss_func = loss_func
        self.centers = centers

    def forward(self, inputs, labels):
        feats, spk = inputs
        loss, _ = self.loss_func(feats, labels, spk=spk, enroll=self.centers, spoofprint=1)
        return loss


if __name__ == '__main__':
    # Prepare your own dataset and data loader
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    cuda = torch.cuda.is_available()
    print('Cuda device available: ', cuda)
    device = torch.device("cuda" if cuda else "cpu")

    with open("aasist/AASIST.conf", "r") as f_json:
        config = json.loads(f_json.read())

    model_config = config["model_config"]
    optim_config = config["optim_config"]
    module = import_module("aasist.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    feat_model = _model(model_config).to(device)

    nb_params = sum([param.view(-1).size()[0] for param in feat_model.parameters()])
    print("no. model params:{}".format(nb_params))
    args = initParams()

    # load datasets
    trainDataLoader, devDataLoader, evalDataLoader, \
    trainBonaLoader, devEnrollLoader, evalEnrollLoader, num_centers = get_loader(args)

    optimizer = torch.optim.Adam(feat_model.parameters(),
                                 lr=0.0001,
                                 betas=optim_config['betas'],
                                 weight_decay=optim_config['weight_decay'],
                                 amsgrad=optim_config['amsgrad'])

    # my loss
    samo = SAMO(160, m_real=0.7, m_fake=0, alpha=20).to(device)
    centers = update_embeds(device, feat_model, trainBonaLoader)

    # Create wrappers
    trainloader_wrapper = MyTrainDataLoaderIter(trainDataLoader)
    model_wrapper = ModelWrapper(feat_model)
    loss_func_wrapper = LossFunctionWrapper(samo, centers)

    # Run LRFinder
    lr_finder = LRFinder(model_wrapper, optimizer, loss_func_wrapper, device='cuda')
    lr_finder.range_test(
        trainloader_wrapper,
        # end_lr=100, num_iter=100,
        end_lr=0.1, num_iter=1103, step_mode='exp', start_lr=1e-7  # for batch_size=23
    )
    lr_finder.plot(suggest_lr=True)
    lr_finder.reset()
