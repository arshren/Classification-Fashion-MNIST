# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:42:15 2020

@author: Renu K
"""
'''
Capirca uses Google's abseil-py library, which uses a Google-specific
    # wrapper for logging. That wrapper will write a warning to sys.stderr if
    # the Google command-line flags library has not been initialized.
    #
    # https://github.com/abseil/abseil-py/blob/pypi-v0.7.1/absl/logging/__init__.py#L819-L825
    #
    # This is not right behavior for Python code that is invoked outside of a
    # Google-authored main program. Use knowledge of abseil-py to disable that
    # warning; ignore and continue if something goes wrong.
    import absl.logging

    # https://github.com/abseil/abseil-py/issues/99
    logging.root.removeHandler(absl.logging._absl_handler)
    # https://github.com/abseil/abseil-py/issues/102
    absl.logging._warn_preinit_stderr = False

'''


from flask import Flask, jsonify, request
from flask_restful import  Api, Resource
import numpy as np
from Fashion_MNIST import classFashionMNIST
from FashionConfig import Config as cfg
import logging
import absl.logging

app=Flask(__name__)


# Load the Class



@app.route("/predict", methods=["GET"])
def predict():
    pred =""
    posted_data = request.get_json()
    test_image_num=posted_data['test_image_num']
    logging.info("In Predict")
    model_filename=cfg.WEIGHT_FILENAME
    pred= fashionMNISTclass.predict_data(test_image_num, model_filename)
    return jsonify(pred)

@app.route("/real", methods=["GET"])
def real():
    data =""
    posted_data = request.get_json()
    test_image_num=posted_data['test_image_num']
    data = fashionMNISTclass.actual_data(test_image_num)
    return jsonify(data)

@app.route("/train", methods=["POST", "GET"])
def train():
    history=""
    posted_data = request.get_json()
    epochs=posted_data['epochs']
    if epochs=="":
        epochs= cfg.EPOCHS
    logging.info("Training ")
    # noramlize the data
    fashionMNISTclass.normalize_data()
    # train the model
    history, model = fashionMNISTclass.train_model(cfg.WEIGHT_FILENAME, 
                                                   epochs,cfg.OPTIMIZER,
                                                   cfg.LEARNING_RATE,
                                                   cfg.BATCH_SIZE) 
    val_acc=str(np.average(history.history['val_acc']))
    acc=str(np.average(history.history['acc']))
    result={'val accuracy':val_acc, 'acc':acc}
    return jsonify(result)

if __name__ == '__main__':
    print("In logging")
     # https://github.com/abseil/abseil-py/issues/99
    logging.root.removeHandler(absl.logging._absl_handler)
    # https://github.com/abseil/abseil-py/issues/102
    absl.logging._warn_preinit_stderr = False
    logging.basicConfig(filename=cfg.LOG_FILENAME, filemode='a', format='%(filename)s-%(asctime)s %(msecs)d- %(process)d-%(levelname)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S %p' ,
                    level=logging.DEBUG)
    fashionMNISTclass= classFashionMNIST(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, cfg.DATA_SIZE, cfg.CLASS_NAME)
    # noramlize the data
    fashionMNISTclass.normalize_data()
    # train the model
    history, model = fashionMNISTclass.train_model(cfg.WEIGHT_FILENAME, 
                                                   cfg.EPOCHS,
                                                   cfg.OPTIMIZER,
                                                   cfg.LEARNING_RATE,
                                                   cfg.BATCH_SIZE) 
    app.run(debug=True)
    