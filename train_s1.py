import os
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sunstage_dataset import SunStageData
from sunstage_model_s1 import SunStage1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_name', type=str, default='dan_1')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--deca_dir', type=str, default='./data/DECA')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_epoch', type=int, default=2000)
    parser.add_argument('--save_dir', type=str, default='./output')
    parser.add_argument('--save_steps', type=int, default=500)
    opt = parser.parse_args()

    os.makedirs(opt.save_dir, exist_ok=True)

    dataset = SunStageData(opt)
    model = SunStage1(opt, len(dataset.camera_dict))
    dataloader = DataLoader(dataset, num_workers=4, batch_size=1, shuffle=True)
    writer = SummaryWriter(f'./runs/{opt.obj_name}_s1')

    for epoch in tqdm(range(opt.n_epoch)):
        loss_epoch = 0.
        for img_dict in dataloader:
            for k in img_dict.keys():
                try:
                    img_dict[k] = img_dict[k].to(opt.device)
                except AttributeError:
                    pass

            silhouette_images, lmk_3d = model.render_s1(img_dict)
            loss = model.step_s1(img_dict, silhouette_images, lmk_3d)
            loss_epoch += loss

        loss_epoch /= len(dataset)
        writer.add_scalar('Loss/total', loss_epoch, epoch)

        if (epoch + 1) % opt.save_steps == 0:
            model.save(opt.save_dir, epoch + 1)
        # print(loss_epoch, n_img)

        #     break
        # break
