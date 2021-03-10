import argparse
import os

from torch.utils.data import DataLoader

from dataset import Dataset
# from dataset import DatasetV2 as Dataset
from model import RFRNetModel


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--mask_root', type=str)
    parser.add_argument('--model_save_path', type=str, default='checkpoint')
    parser.add_argument('--result_save_path', type=str, default='results')
    parser.add_argument('--target_size', type=int, default=256)
    parser.add_argument('--mask_mode', type=int, default=1)
    parser.add_argument('--num_iters', type=int, default=450000)
    parser.add_argument('--model_path', type=str,
                        default="checkpoint/100000.pth")
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--n_threads', type=int, default=6)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()

    model = RFRNetModel()
    if args.test:
        model.initialize_model(args.model_path, False)
        model.cuda()

        dataloader = DataLoader(Dataset(args.data_root, args.mask_root,
                                        args.mask_mode,
                                        args.target_size,
                                        mask_reverse=True,
                                        training=False))

        model.test(dataloader, args.result_save_path)
    else:
        model.initialize_model(args.model_path, True)
        model.cuda()

        dataloader = DataLoader(Dataset(args.data_root, args.mask_root,
                                        args.mask_mode,
                                        args.target_size,
                                        mask_reverse=True),
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.n_threads,
                                pin_memory=True)

        model.train(dataloader, args.model_save_path,
                    args.finetune, args.num_iters,
                    multi_gpu=args.multi_gpu,
                    fp16=args.fp16)


if __name__ == '__main__':
    run()
