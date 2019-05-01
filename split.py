#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Import necessaries libraries
import argparse
import os
import random
import shutil

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", required=False, type=str,
                default="jpg", help="Choose type of your files")
ap.add_argument("-p", "--percent", required=False, type=float,
                default=0.2, help="Choose percent to split")
ap.add_argument("-ts", "--test", required=False, type=str,
                default="test", help="Choose your test directory")
ap.add_argument("-tr", "--train", required=False, type=str,
                default="train", help="Choose your train directory")
args = vars(ap.parse_args())
# Get list with all files
all_files = next(os.walk(os.getcwd() + "/images"))[2]
# Remain only files with specified type
files_with_type = [f for f in all_files if "." + args["type"] in f]
# Create a number to split based on percent
num_to_select = int(len(files_with_type)*args["percent"])
# Split into train and test
list_to_test = random.sample(files_with_type, num_to_select)
list_to_train = [f for f in files_with_type if f not in list_to_test]
# Create directories if not exists
if not os.path.exists("images/" + args["test"]):
    os.makedirs("images/" + args["test"])
if not os.path.exists("images/" + args["train"]):
    os.makedirs("images/" + args["train"])
# Copy selected files to directories
for f in list_to_test:
    supfile = f.replace(".jpg", ".xml")
    shutil.copy("images/" + f, "images/" + args["test"] + "/" + f)
    if "images/" + supfile is not None:
        shutil.copy("images/" + supfile, "images/" + args["test"] + "/" + supfile)
for f in list_to_train:
    supfile = f.replace(".jpg", ".xml")
    shutil.copy("images/" + f, "images/" + args["train"] + "/" + f)
    if "images/" + supfile is not None:
        shutil.copy("images/" + supfile, "images/" + args["train"] + "/" + supfile)
