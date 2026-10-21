#! /usr/bin/env python
import os, sys
try:
    os.chdir('/ThermalGAN/src/scripts')
except:
     os.chdir('./')
currentdir = os.path.dirname(os.getcwd())

print(os.listdir(os.getcwd()))
print(os.listdir(currentdir))
print(os.listdir(os.path.dirname(currentdir)))
sys.path.append(currentdir)


import yaml
import pandas as pd
import tensorflow as tf

from utils import models_classifyer as models_class
from utils import preprocessing as pre

def main():
    print(f"Tf version is {tf.__version__}")
    os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"


    config = {"Data_meso":{
    "max_length": 512,
    "max_samples": 4000000,
    "base_dir": '/data/records',
    "train_dir": 'OGT_IMG_train/Meso_*.tfrecord',
    "val_dir": 'OGT_IMG_test/Meso_*.tfrecord',
    "n_shards": 1000, # How many shards to use
    "shards": 100},
    
"Data_thermo":{
    "max_length": 512,
    "max_samples": 4000000,
    "base_dir": '/data/records',
    "train_dir": 'OGT_IMG_train/Target_*.tfrecord',
    "val_dir": 'OGT_IMG_test/Target_*.tfrecord',
    "n_shards": 1000, # How many shards to use
    "shards": 100}}
    

    data_train_meso, data_val_meso = pre.load_data_bert_inference(config["Data_meso"])
    data_train_thermo, data_val_thermo = pre.load_data_bert_inference(config["Data_thermo"])

    data = {'meso_train': data_train_meso, 'thermo_train': data_train_thermo, 'meso_val':data_val_meso , 'thermo_val': data_val_thermo}

    print("Loaded data")


    models_weights = ["../../weights/OGT/Model1/variables/variables", "../../weights/OGT/Model2/variables/variables", "../../weights/OGT/Model3/variables/variables"]

    file = "../../config/Classifier/config_classifier1.yaml"
    with open(file, 'r') as file_descriptor:
        config_class = yaml.load(file_descriptor, Loader=yaml.FullLoader)



    model_input = tf.keras.layers.Input(shape=(512,21))
    model1 = models_class.get_classifier(config_class['Classifier'], 21)
    model2 = models_class.get_classifier(config_class['Classifier'], 21)
    model3 = models_class.get_classifier(config_class['Classifier'], 21)

    print("Initialized regression models")

    output1 = model1(model_input)
    output2 = model2(model_input)
    output3 = model3(model_input)


    model1.load_weights(models_weights[0]).expect_partial()
    model2.load_weights(models_weights[1]).expect_partial()
    model3.load_weights(models_weights[2]).expect_partial()

    print("Loaded weights regression model")

    ensemble_output = tf.keras.layers.Average()([output1, output2, output3])
    ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output)

    
    batches_x = data['meso_train'].batch(32768, drop_remainder=False) 
    batches_y = data['thermo_train'].batch(32768, drop_remainder=False)

        
    #Anneal schedule for gumbel

    # get thermophiles
    ids       = []
    true_temps = []
    pred_temps = []

# Predicting Thermophiles
    for step, x in enumerate(batches_y):
        seqs      = x[0]
        id_       = x[-1]
        true_temp = x[1]
        pred_temp = ensemble_model.predict(seqs)
        
        ids += id_.numpy().tolist()
        tmp = true_temp.numpy() + 50.5
        true_temps += tmp.tolist()
        pred_temps += [i[0] for i in pred_temp.tolist()]
    print(f"Running predictions on Mesophiles for {step} steps")
        # Write history obj
    df = pd.DataFrame({"id":ids, "Temperature":true_temps, "Predicted Temperature": pred_temps})
    df.to_csv(os.path.join("/results",'Temperature_OGT_IMG_Mesophiles.csv'))
#predicting mesophiles
    for step, x in enumerate(batches_x):
        seqs      = x[0]
        id_       = x[-1]
        true_temp = x[1]
        pred_temp = ensemble_model.predict(seqs)
        
        ids += id_.numpy().tolist()
        tmp = true_temp.numpy() + 50.5
        true_temps += tmp.tolist()
        pred_temps += [i[0] for i in pred_temp.tolist()]

    print(f"Running predictions on Thermophiles for {step} steps")

    # Write history obj
    df = pd.DataFrame({"id":ids, "Temperature":true_temps, "Predicted Temperature": pred_temps})
    df.to_csv(os.path.join("/results",'Temperature_OGT_IMG_Thermophiles.csv'))
        
    return 0
    
if __name__ == "__main__":
    main()