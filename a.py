from flask import Flask, render_template, request
import json
import os
import urllib.request
import numpy as np
import pathlib
from fastai.text.all import *
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

learner = load_learner('stance_prediction_model.pkl')
prediction = learner.predict(
    "Please tweet something regarding Indian farmers as they are ignored by Indian government and national media so that their voice can’t be seen by the world, A peaceful protest going on by Indian farmers from nearly 80 days at capital borders, you can see about it  #FarmersProtest")[0]
print(prediction)
