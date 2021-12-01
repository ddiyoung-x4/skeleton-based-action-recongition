import pandas
import numpy as np

def data_augmentation(raw_data):
    # Left -> Right, Right -> Left
    df = raw_data[['nose_x', 'nose_y', 'neck_x', 'neck_y', 'LShoulder_x', 'LShoulder_y', 'LElbow_x', 'LElbow_y', 'LWrist_x', 'LWrist_y',
        'Rshoulder_x', 'Rshoulder_y', 'Relbow_x', 'Relbow_y', 'Rwrist_x', 'Rwrist_y', 'LHip_x', 'LHip_y', 'LKnee_x', 'LKnee_y', 'LAnkle_x', 'LAnkle_y', 
        'RHip_x', 'RHip_y', 'RKnee_x', 'RKnee_y', 'RAnkle_x', 'RAnkle_y', 'LEye_x', 'LEye_y', 'REye_x', 'REye_y', 'LEar_x', 'LEar_y', 'REar_x', 'REar_y',
        'class']]
    dataset = df.values
    new_dataset = dataset.copy()
    
    # (x, y) -> (1 - x, y)
    for i in range(0, 18):
        new_dataset[:, 2*i] = 1 - dataset[:, 2*i]
    
    # concatenate
    res = np.concatenate((dataset, new_dataset), axis=0)
    
    return res
    
