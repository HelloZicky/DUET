# coding=utf-8
import os
import time
import json
import logging
import math
import argparse
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.multiprocessing as mp
from torch import nn

import model
from util.timer import Timer

from util import args_processing as ap
from util import consts
from util import env

from loader import multi_metric_meta_sequence_dataloader as sequence_dataloader

from util import new_metrics
import numpy as np
from thop import profile

from util import utils
utils.setup_seed(0)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=str, help="Kernels configuration for CNN")
    parser.add_argument("--bucket", type=str, default=None, help="Bucket name for external storage")
    parser.add_argument("--dataset", type=str, default="alipay", help="Bucket name for external storage")

    parser.add_argument("--max_steps", type=int, help="Number of iterations before stopping")
    parser.add_argument("--snapshot", type=int, help="Number of iterations to dump model")
    parser.add_argument("--checkpoint_dir", type=str, help="Path of the checkpoint path")
    parser.add_argument("--learning_rate", type=str, default=0.001)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")

    parser.add_argument("--max_epoch", type=int, default=1, help="Max epoch")
    parser.add_argument("--num_loading_workers", type=int, default=4, help="Number of threads for loading")
    parser.add_argument("--model", type=str, help="model type")
    parser.add_argument("--init_checkpoint", type=str, default="", help="Path of the checkpoint path")
    parser.add_argument("--init_step", type=int, default=0, help="Path of the checkpoint path")

    parser.add_argument("--max_gradient_norm", type=float, default=0.)

    parser.add_argument("--arch_config_path", type=str, default=None, help="Path of model configs")
    parser.add_argument("--arch_config", type=str, default=None, help="base64-encoded model configs")

    return parser.parse_known_args()[0]


def predict(predict_dataset, model_obj, device, args, train_epoch, train_step, y_list_overall, pred_list_overall, writer=None, buffer_overall=None):
    model_obj.eval()
    model_obj.to(device)

    timer = Timer()
    log_every = 200

    pred_list = []
    y_list = []

    for step, batch_data in enumerate(predict_dataset, 1):
        logits = model_obj({
            key: value.to(device)
            for key, value in batch_data.items()
            if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL, consts.FIELD_TRIGGER_SEQUENCE}
        })

        prob = torch.sigmoid(logits).detach().cpu().numpy()
        y = batch_data[consts.FIELD_LABEL].view(-1, 1)
        auc, _, _, _ = new_metrics.calculate_overall_auc(prob, y)
        ndcg = 0
        pred_list.extend(prob)
        y_list.extend(np.array(y))

        buffer_overall.extend(
            [int(user_id), float(score), float(label)]
            for user_id, score, label
            in zip(
                batch_data[consts.FIELD_USER_ID],
                prob,
                batch_data[consts.FIELD_LABEL]
            )
        )

        if step % log_every == 0:
            logger.info(
                "train_epoch={}, step={}, auc={:5f}, ndcg={:5f}, speed={:2f} steps/s".format(
                    train_epoch, step, auc, ndcg, log_every / timer.tick(False)
                )
            )

    auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred_list), np.array(y_list))
    ndcg = 0

    print("train_epoch={}, train_step={}, auc={:5f}, ndcg={:5f}".format(train_epoch, train_step, auc, ndcg))
    with open(os.path.join(args.checkpoint_dir, "log_ood.txt"), "a") as writer:
        print("train_epoch={}, train_step={}, auc={:5f}, ndcg={:5f}".format(train_epoch, train_step, auc, ndcg), file=writer)

    return auc, pred_list, y_list


def train(train_dataset, model_obj, device, args, pred_dataloader, y_list_overall, pred_list_overall, is_finetune, buffer_overall):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model_obj.parameters(),
        lr=float(args.learning_rate)
    )
    model_obj.train()
    model_obj.to(device)

    logger.info("Start training...")
    timer = Timer()
    log_every = 200

    max_step = 0
    best_auc = 0
    best_ckpt_path = os.path.join(args.checkpoint_dir, "best_auc" + ".pkl")
    for epoch in range(1, args.max_epoch + 1):
        for step, batch_data in enumerate(train_dataset, 1):
            logits = model_obj({
                key: value.to(device)
                for key, value in batch_data.items()
                if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL, consts.FIELD_TRIGGER_SEQUENCE}
            })

            loss = criterion(logits, batch_data[consts.FIELD_LABEL].view(-1, 1).to(device))
            pred, y = torch.sigmoid(logits), batch_data[consts.FIELD_LABEL].view(-1, 1)

            auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred.detach().cpu()), np.array(y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % log_every == 0:
                logger.info(
                    "epoch={}, step={}, loss={:5f}, auc={:5f}, speed={:2f} steps/s".format(
                        epoch, step, float(loss.item()), auc, log_every / timer.tick(False)
                    )
                )
            max_step = step

        pred_auc, pred_list, y_list = predict(
            predict_dataset=pred_dataloader,
            model_obj=model_obj,
            device=device,
            args=args,

            train_epoch=epoch,
            train_step=epoch * max_step,
            y_list_overall=y_list_overall,
            pred_list_overall=pred_list_overall,
            buffer_overall=buffer_overall
        )

        logger.info("dump checkpoint for epoch {}".format(epoch))
        model_obj.train()
        if pred_auc > best_auc:
            best_auc = pred_auc
            torch.save(model_obj, best_ckpt_path)

    return pred_list, y_list


def load_model(args, model_obj):
    ckpt_path = os.path.join(args.checkpoint_dir, "base", "best_auc.pkl")

    model_load = torch.load(ckpt_path)
    model_load_name_set = set()
    for name, parms in model_load.named_parameters():
        model_load_name_set.add(name)
    print(model_load_name_set)
    model_load_dict = model_load.state_dict()
    model_obj_dict = model_obj.state_dict()
    model_obj_dict.update(model_load_dict)
    model_obj.load_state_dict(model_obj_dict)

    return model_obj, model_load_name_set


def main_worker(_):
    args = parse_args()
    ap.print_arguments(args)
    model_meta = model.get_model_meta(args.model)  # type: model.ModelMeta
    model_conf, raw_model_conf = ap.parse_arch_config_from_args(model_meta, args)  # type: dict

    # Construct model
    model_obj = model_meta.model_builder(model_conf=model_conf)  # type: torch.nn.module
    model_obj, model_load_name_set = load_model(args, model_obj)
    import copy
    model_obj_backup = copy.deepcopy(model_obj)

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, "base_finetune")
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    print("=" * 100)
    for name, parms in model_obj.named_parameters():
        print(name)
    print("=" * 100)
    device = env.get_device()

    worker_id = worker_count = 8
    y_list_overall = []
    pred_list_overall = []
    buffer_overall = []
    finetune_dataset_dir = os.path.join(args.dataset, "finetune_dataset")
    print(model_obj)

    user_folder_name_list = os.listdir(finetune_dataset_dir)
    for index, user_folder_name in enumerate(user_folder_name_list, 1):
        print(index, len(user_folder_name_list), sep="/")
        model_obj = copy.deepcopy(model_obj_backup)
        train_file = os.path.join(finetune_dataset_dir, user_folder_name, "train.txt")
        if not os.path.exists(train_file):
            is_finetune = False
        else:
            is_finetune = True
        test_file = os.path.join(finetune_dataset_dir, user_folder_name, "test.txt")

        args.num_loading_workers = 1

        # Setup up data loader
        pred_dataloader = sequence_dataloader.MetaSequenceDataLoader(
            table_name=test_file,
            slice_id=args.num_loading_workers * worker_id,
            slice_count=args.num_loading_workers * worker_count,
            is_train=False
        )
        pred_dataloader = torch.utils.data.DataLoader(
            dataset=pred_dataloader,
            batch_size=args.batch_size,
            num_workers=args.num_loading_workers,
            pin_memory=True,
            collate_fn=pred_dataloader.batchify,
            drop_last=False,
        )

        if is_finetune:
            train_dataloader = sequence_dataloader.MetaSequenceDataLoader(
                table_name=train_file,
                slice_id=0,
                slice_count=args.num_loading_workers,
                is_train=True
            )
            train_dataloader = torch.utils.data.DataLoader(
                dataset=train_dataloader,
                batch_size=args.batch_size,
                num_workers=args.num_loading_workers,
                pin_memory=True,
                collate_fn=train_dataloader.batchify,
                drop_last=False,
            )

            # Setup training
            pred_list, y_list = train(
                train_dataset=train_dataloader,
                model_obj=model_obj,
                device=device,
                args=args,
                pred_dataloader=pred_dataloader,
                y_list_overall=y_list_overall,
                pred_list_overall=pred_list_overall,
                is_finetune=is_finetune,
                buffer_overall=buffer_overall
            )
        else:
            pred_auc, pred_list, y_list = predict(
                predict_dataset=pred_dataloader,
                model_obj=model_obj,
                device=device,
                args=args,
                train_epoch=0,
                train_step=0,
                y_list_overall=y_list_overall,
                pred_list_overall=pred_list_overall,
                buffer_overall=buffer_overall
            )

        pred_list_overall.extend(pred_list)
        y_list_overall.extend(y_list)

    print(len(pred_list_overall))
    print(len(y_list_overall))

    overall_auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred_list_overall), np.array(y_list_overall))
    user_auc = new_metrics.calculate_user_auc(buffer_overall)
    overall_logloss = new_metrics.calculate_overall_logloss(np.array(pred_list_overall), np.array(y_list_overall))
    user_ndcg5, user_hr5 = new_metrics.calculate_user_ndcg_hr(5, buffer_overall)
    user_ndcg10, user_hr10 = new_metrics.calculate_user_ndcg_hr(10, buffer_overall)
    user_ndcg20, user_hr20 = new_metrics.calculate_user_ndcg_hr(20, buffer_overall)

    print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
          "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
          format(1, 1, overall_auc, user_auc, overall_logloss,
                 user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20))
    with open(os.path.join(args.checkpoint_dir, "log_overall.txt"), "a") as writer:
        print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
              "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
              format(1, 1, overall_auc, user_auc, overall_logloss,
                     user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20), file=writer)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    mp.spawn(main_worker, nprocs=1)

