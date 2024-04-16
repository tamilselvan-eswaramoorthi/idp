import os
import time
import torch
import torch.optim as optim

from text_detector.model import CRAFT
from text_detector.trainer.config.load_config import DotDict
from text_detector.trainer.data.dataset import CustomDataset
from text_detector.trainer.loss.mseloss import Maploss_v2, Maploss_v3
from text_detector.trainer.metrics.eval_det_iou import DetectionIoUEvaluator

from text_detector.trainer.eval import main_eval
from text_detector.utils.general import copyStateDict


class Trainer(object):
    def __init__(self, config, device, mode):
        self.device = device
        self.config = config
        self.mode = mode
        if self.config.train.ckpt_path is not None:
            self.net_param = torch.load(self.config.train.ckpt_path, map_location=self.device)
        else:
            self.net_param = None

    def get_custom_dataset(self):
        custom_dataset = CustomDataset(
            output_size=self.config.train.data.output_size,
            data_dir=self.config.data_root_dir,
            saved_gt_dir=None,
            mean=self.config.train.data.mean,
            variance=self.config.train.data.variance,
            gauss_init_size=self.config.train.data.gauss_init_size,
            gauss_sigma=self.config.train.data.gauss_sigma,
            enlarge_region=self.config.train.data.enlarge_region,
            enlarge_affinity=self.config.train.data.enlarge_affinity,
            watershed_param=self.config.train.data.watershed,
            aug=self.config.train.data.custom_aug,
            vis_test_dir=self.config.vis_test_dir,
            sample=self.config.train.data.custom_sample,
            vis_opt=self.config.train.data.vis_opt,
            pseudo_vis_opt=self.config.train.data.pseudo_vis_opt,
            do_not_care_label=self.config.train.data.do_not_care_label,
        )

        return custom_dataset

    def adjust_learning_rate(self, optimizer, gamma, step, lr):
        lr = lr * (gamma ** step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return param_group["lr"]

    def get_loss(self):
        if self.config.train.loss == 2:
            criterion = Maploss_v2()
        elif self.config.train.loss == 3:
            criterion = Maploss_v3()
        else:
            raise Exception("Undefined loss")
        return criterion

    def iou_eval(self, dataset, train_step, buffer, model):
        test_config = DotDict(self.config.test[dataset])

        val_result_dir = os.path.join(
            self.config.results_dir, "{}/{}".format(dataset + "_iou", str(train_step))
        )

        evaluator = DetectionIoUEvaluator()

        metrics = main_eval(
            None,
            self.config.train.backbone,
            test_config,
            evaluator,
            val_result_dir,
            buffer,
            model,
            self.mode,
        )

    def train(self, buffer_dict):
        
        if self.device != 'cpu':
            torch.cuda.set_device(int(self.device.replace('cuda:', '')))

        # MODEL -------------------------------------------------------------------------------------------------------#
        # SUPERVISION model
        if self.config.mode == "weak_supervision":
            if self.config.train.backbone == "vgg":
                supervision_model = CRAFT(pretrained=False, amp=self.config.train.amp)
            else:
                raise Exception("Undefined architecture")

            supervision_device = self.device
            if self.config.train.ckpt_path is not None:
                supervision_param = torch.load(self.config.train.ckpt_path, map_location=self.device)
                supervision_model.load_state_dict(copyStateDict(supervision_param["craft"]))
                supervision_model = supervision_model.to(self.device)
        else:
            supervision_model, supervision_device = None, None

        # TRAIN model
        if self.config.train.backbone == "vgg":
            craft = CRAFT(pretrained=False, amp=self.config.train.amp)
        else:
            raise Exception("Undefined architecture")

        if self.config.train.ckpt_path is not None:
            craft.load_state_dict(copyStateDict(self.net_param["craft"]))

        if self.device != 'cpu':
            craft = craft.cuda()
            craft = torch.nn.DataParallel(craft)
            torch.backends.cudnn.benchmark = True

        # DATASET -----------------------------------------------------------------------------------------------------#


        trn_real_dataset = self.get_custom_dataset()


        if self.config.mode == "weak_supervision":
            trn_real_dataset.update_model(supervision_model)
            trn_real_dataset.update_device(supervision_device)

        trn_real_loader = torch.utils.data.DataLoader(
            trn_real_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            drop_last=False,
            pin_memory=True,
        )

        # OPTIMIZER ---------------------------------------------------------------------------------------------------#
        optimizer = optim.Adam(
            craft.parameters(),
            lr=self.config.train.lr,
            weight_decay=self.config.train.weight_decay,
        )

        if self.config.train.ckpt_path is not None and self.config.train.st_iter != 0:
            optimizer.load_state_dict(copyStateDict(self.net_param["optimizer"]))
            self.config.train.st_iter = self.net_param["optimizer"]["state"][0]["step"]
            self.config.train.lr = self.net_param["optimizer"]["param_groups"][0]["lr"]

        # LOSS --------------------------------------------------------------------------------------------------------#
        # mixed precision
        if self.config.train.amp:
            scaler = torch.cuda.amp.GradScaler()

            if (
                    self.config.train.ckpt_path is not None
                    and self.config.train.st_iter != 0
            ):
                scaler.load_state_dict(copyStateDict(self.net_param["scaler"]))
        else:
            scaler = None

        criterion = self.get_loss()

        # TRAIN -------------------------------------------------------------------------------------------------------#
        train_step = self.config.train.st_iter
        whole_training_step = self.config.train.end_iter
        update_lr_rate_step = 0
        training_lr = self.config.train.lr
        loss_value = 0
        batch_time = 0
        start_time = time.time()

        print(
            "================================ Train start ================================"
        )
        while train_step < whole_training_step:
            for (
                    index,
                    (
                            images,
                            region_scores,
                            affinity_scores,
                            confidence_masks,
                    ),
            ) in enumerate(trn_real_loader):
                craft.train()
                if train_step > 0 and train_step % self.config.train.lr_decay == 0:
                    update_lr_rate_step += 1
                    training_lr = self.adjust_learning_rate(
                        optimizer,
                        self.config.train.gamma,
                        update_lr_rate_step,
                        self.config.train.lr,
                    )

                images = images.cuda(non_blocking=True)
                region_scores = region_scores.cuda(non_blocking=True)
                affinity_scores = affinity_scores.cuda(non_blocking=True)
                confidence_masks = confidence_masks.cuda(non_blocking=True)


                region_image_label = region_scores
                affinity_image_label = affinity_scores
                confidence_mask_label = confidence_masks

                if self.config.train.amp:
                    with torch.cuda.amp.autocast():

                        output, _ = craft(images)
                        out1 = output[:, :, :, 0]
                        out2 = output[:, :, :, 1]

                        loss = criterion(
                            region_image_label,
                            affinity_image_label,
                            out1,
                            out2,
                            confidence_mask_label,
                            self.config.train.neg_rto,
                            self.config.train.n_min_neg,
                        )

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    output, _ = craft(images)
                    out1 = output[:, :, :, 0]
                    out2 = output[:, :, :, 1]
                    loss = criterion(
                        region_image_label,
                        affinity_image_label,
                        out1,
                        out2,
                        confidence_mask_label,
                        self.config.train.neg_rto,
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                end_time = time.time()
                loss_value += loss.item()
                batch_time += end_time - start_time

                if train_step > 0 and train_step % 5 == 0:
                    mean_loss = loss_value / 5
                    loss_value = 0
                    avg_batch_time = batch_time / 5
                    batch_time = 0

                    print(
                        "{}, training_step: {}|{}, learning rate: {:.8f}, "
                        "training_loss: {:.5f}, avg_batch_time: {:.5f}".format(
                            time.strftime(
                                "%Y-%m-%d:%H:%M:%S", time.localtime(time.time())
                            ),
                            train_step,
                            whole_training_step,
                            training_lr,
                            mean_loss,
                            avg_batch_time,
                        )
                    )

                if (
                        train_step % self.config.train.eval_interval == 0
                        and train_step != 0
                ):

                    craft.eval()

                    print("Saving state, index:", train_step)
                    save_param_dic = {
                        "iter": train_step,
                        "craft": craft.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    save_param_path = (
                            self.config.results_dir
                            + "/CRAFT_clr_"
                            + repr(train_step)
                            + ".pth"
                    )

                    if self.config.train.amp:
                        save_param_dic["scaler"] = scaler.state_dict()
                        save_param_path = (
                                self.config.results_dir
                                + "/CRAFT_clr_amp_"
                                + repr(train_step)
                                + ".pth"
                        )

                    torch.save(save_param_dic, save_param_path)

                    # validation
                    self.iou_eval(
                        "custom_data",
                        train_step,
                        buffer_dict["custom_data"],
                        craft,
                    )

                train_step += 1
                if train_step >= whole_training_step:
                    break

            if self.config.mode == "weak_supervision":
                state_dict = craft.module.state_dict()
                supervision_model.load_state_dict(state_dict)
                trn_real_dataset.update_model(supervision_model)

        # save last model
        save_param_dic = {
            "iter": train_step,
            "craft": craft.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_param_path = (
                self.config.results_dir + "/CRAFT_clr_" + repr(train_step) + ".pth"
        )

        if self.config.train.amp:
            save_param_dic["scaler"] = scaler.state_dict()
            save_param_path = (
                    self.config.results_dir
                    + "/CRAFT_clr_amp_"
                    + repr(train_step)
                    + ".pth"
            )
        torch.save(save_param_dic, save_param_path)

