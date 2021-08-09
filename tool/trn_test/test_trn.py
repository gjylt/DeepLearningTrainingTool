import json
import os
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from model.TRN.dataset import customCVSDataSet
from model.TRN.models import TSN
from model.TRN.transforms import *
from model.TRN.ops.opts import parser

import tqdm

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    check_rootfolders()

    # Phase
    dictCategories = {'phase': [ '7', '8', '9', '10']}
    dictCategories = {'phase': ['0', '1', '2', '3','4','5','6']}
    rule = 'phase'

    dirImage   = '/home/withai/Pictures/LCFrame/append_video-8fps'
    pathJson   = '/home/withai/Desktop/LCLabelFiles/LCPhase_222_len24_2_annotator_test_checked.json'
    recordJson = "/home/withai/Desktop/LCLabelFiles/LCPhase_222_len24_2_annotator_test_result.json"

    categories = dictCategories[rule]
    num_class  = len(categories)
    # args.resume = f"/home/withai/wangyx/checkPoint/TRN/{rule}.pth.tar"
    args.resume = '/home/withai/Desktop/LCLabelFiles/TRN_6phase_222_2anitator_best.pth.tar'

    args.store_name = '_'.join(
        ['TRN', args.dataset, args.modality, args.arch, args.consensus_type, 'segment%d' % args.num_segments])
    print('storing name: ' + args.store_name)

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn)

    crop_size  = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std  = model.input_std

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    if os.path.isfile(args.resume):
        print(("=> loading checkpoint '{}'".format(args.resume)))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print(("=> loaded checkpoint '{}' (epoch {})"
               .format(args.evaluate, checkpoint['epoch'])))
    else:
        print(("=> no checkpoint found at '{}'".format(args.resume)))
        exit()

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    val_loader = torch.utils.data.DataLoader(
        customCVSDataSet(dirImage,
                         pathJson,
                         'test', rule, torchvision.transforms.Compose([
                GroupScale(int(scale_size)),
                GroupCenterCrop(crop_size),
                Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                normalize,
            ]), args.num_segments, dictCategories[rule]),
        batch_size=8, shuffle=False,
        num_workers=32, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    prec1 = validate(val_loader, model, criterion, recordJson)


def validate(val_loader, model, criterion, recordJson):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    calLabel = []
    calResult = []
    calConf = []
    dictOutput = {}
    for i, (input, target, imagesName) in enumerate(tqdm.tqdm(val_loader)):
        target = target.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        (prec1, prec5), targetTemp, resultTemp, confTemp = accuracy(output.data, target, topk=(1, 3))
        calLabel.append(targetTemp)
        calResult.append(resultTemp)
        calConf.append(confTemp)

        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        for num, imageName in enumerate(imagesName):
            imageName = imageName.split('.')[0]
            dictOutput[imageName] = [resultTemp[num].item(),
                                     targetTemp[num].item(),
                                     confTemp[num].item()]

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output)
    output_best = '\nBest Prec@1: %.3f' % (best_prec1)
    print(output_best)
    print(dictOutput)
    with open(recordJson, "w", encoding='utf-8') as f:
        json.dump(dictOutput, f, ensure_ascii=False, indent=2)
        f.close()

    labels = torch.cat(calLabel, 0).cpu().detach()
    results = torch.cat(calResult, 0).cpu().detach()
    conf = torch.cat(calConf, 0).cpu().detach()
    tp = (labels == results).long()
    getConfusionMatrix(tp.numpy(), labels.numpy(), results.numpy(), conf.numpy())

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    output = torch.nn.Softmax(1)(output)

    conf, pred = output.topk(maxk, 1, True, True)

    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred[0], conf[:, 0]


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model, args.root_output]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


from tool.metric.calculateAP import ap_per_class


def getConfusionMatrix(tp, labels, results, conf):
    p, r, ap, f1, classes = ap_per_class(tp, conf, results, labels)
    print(f"{classes.tolist()}\n{p.tolist()}\n{r.tolist()}\n{ap.tolist()}\n")


if __name__ == '__main__':

    gpuid = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    main()
