import os
import torch
import torch.nn.parallel
import torch.optim

from model.TRN.models import TSN
from model.TRN.ops.opts import parser

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    # Phase
    dictCategories = {'phase': ['0', '1', '2', '3', '4']}
    rule = 'phase'
    categories = dictCategories[rule]
    num_class = 7 #len(categories)

    args.resume = r"/home/withai/Desktop/LCLabelFiles/TRN_something_RGB_BNInception_TRNmultiscale_segment8_6phaseBgNoMoreThanTarget_best.pth.tar"
    save_path = r"/home/withai/Desktop/LCLabelFiles/TRN_something_RGB_BNInception_TRNmultiscale_segment8_6phaseBgNoMoreThanTarget_best.pth"


    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn)

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

    torch.save(model, save_path)




if __name__ == '__main__':
    gpuid = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    main()
