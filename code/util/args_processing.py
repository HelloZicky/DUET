# -*- coding: utf-8 -*-
"""
Argument parsing and dumpling
"""
import copy
import json
import base64
import os.path
from argparse import Namespace

_ARCH_CONFIG_FILE_NAME = "arch_conf.json"
_TRAIN_ARGS_FILE_NAME = "train_args.json"


def parse_arch_config_from_args(model, args):
    """
    Read or parse arch config
    :param model:
    :param args:
    :return:
    """
    if args.arch_config is not None:
        with open(args.arch_config) as jsonfile:
            raw_arch_config = json.load(jsonfile)
    elif args.arch_config_path is not None:
        with open(args.arch_config_path, "rt") as reader:
            raw_arch_config = json.load(reader)
    else:
        raise KeyError("Model configuration not found")

    return model.arch_config_parser(raw_arch_config), raw_arch_config



def parse_arch_config_from_args_get_profile(model_meta, arch_config, bucket):
    """
    Read or parse arch config
    :param model_meta:
    :param args:
    :return:
    """
    print(arch_config)
    # raw_arch_config = json.loads(arch_config)
    f = open(arch_config, encoding="utf-8")
    raw_arch_config = json.load(f)

    return model_meta.arch_config_parser(raw_arch_config), raw_arch_config


def dump_config(checkpoint_dir, file_name, config_obj):
    """
    Dump configurations to OSS
    :param checkpoint_dir:
    :param file_name:
    :param config_obj:
    :return:
    """
    print(config_obj, file=open(os.path.join(checkpoint_dir, file_name), "w+"))


# def dump_model_config(checkpoint_dir, raw_model_arch, bucket):
def dump_model_config(checkpoint_dir, raw_model_arch):
    """
    Dump model configurations to OSS
    :param args: Namespace object, parsed from command-line arguments
    :param raw_model_arch:
    :return:
    """
    dump_config(checkpoint_dir, _ARCH_CONFIG_FILE_NAME, raw_model_arch)


# def dump_train_arguments(checkpoint_dir, args, bucket):
def dump_train_arguments(checkpoint_dir, args):
    args_dict = copy.copy(args.__dict__)
    args_dict.pop("arch_config")
    dump_config(checkpoint_dir, _TRAIN_ARGS_FILE_NAME, args_dict)


def print_arguments(args):
    print("\nMain arguments:")
    for k, v in args.__dict__.items():
        print("{}={}".format(k, v))
