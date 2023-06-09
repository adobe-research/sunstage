import os
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sunstage_dataset import SunStageData
from sunstage_model_s2 import SunStage2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_name', type=str, default='dan_1')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--deca_dir', type=str, default='./data/DECA')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_epoch', type=int, default=2000)
    parser.add_argument('--s1_dir', type=str, default='./output')
    parser.add_argument('--s1_epoch', type=int, default=2000)
    parser.add_argument('--lr_decay', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='./output')
    parser.add_argument('--save_steps', type=int, default=500)
    opt = parser.parse_args()

    os.makedirs(opt.save_dir, exist_ok=True)

    dataset = SunStageData(opt)
    model = SunStage2(opt, len(dataset.camera_dict))
    dataloader = DataLoader(dataset, num_workers=4, batch_size=1, shuffle=True)
    writer = SummaryWriter(f'./runs/{opt.obj_name}_s2')

    for epoch in tqdm(range(opt.n_epoch)):
        loss_epoch = 0.
        n_mv_full = 0
        for img_dict in dataloader:
            for k in img_dict.keys():
                try:
                    img_dict[k] = img_dict[k].to(opt.device)
                except AttributeError:
                    pass

            full_loss = True
            if not img_dict['full_lmk']:
                if n_mv_full > dataset.n_mv_full:
                    full_loss = False
                n_mv_full += 1

            rendering = model.render_s2(img_dict)
            loss = model.step_s2(img_dict, rendering, full_loss)
            loss_epoch += loss

            if img_dict['img_id'][0] == '0019':
                writer.add_image('ours', rendering[0][0], epoch)
                writer.add_image('gt', img_dict['img_gt'][0][0], epoch)
            # break

        loss_epoch /= len(dataset)
        writer.add_scalar('Loss/total', loss_epoch, epoch)

        if (epoch + 1) % opt.save_steps == 0:
            model.save(opt.save_dir, epoch + 1)
        if (epoch + 1) % opt.lr_decay == 0:
            for param_group in model.optimizer.param_groups:
                param_group['lr'] *= .1


        # print(loss_epoch, len(dataset))
        # break
