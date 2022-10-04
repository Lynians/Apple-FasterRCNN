import os
import numpy as np
import pandas as pd
import torch

def get_csv_data(csv_file_path, train_folder, val_folder, resize_shape=None):
    """_summary_
        csv 파일을 읽어서 filepath, box coordinate 정보 반환
        (주의) train_folder과 val_folder에 같은 파일이 없다고 가정함
    Args:
        csv_file_path (_type_): csv 파일 경로
        train_folder (_type_): train 이미지가 있는 폴더 경로
        val_folder (_type_): validation 이미지가 있는 폴더 경로
        resize_shape (_type_): 입력 이미지를 resize 하는 크기

    Returns:
        _type_: _description_
    """
    
    # 파일 읽어들이기
    csv_file = pd.read_csv(csv_file_path)
    train_files = os.listdir(train_folder)
    val_files = os.listdir(val_folder)
    
    folders = {'train': train_folder, 'val': val_folder}
    files = {'train': train_files, 'val': val_files}
    
    file_paths = {'train': [], 'val': []}    
    box_coords_and_labels = {'train': [], 'val': []}
    filename = ''
    prev_filename = ''
    
    box_coords = []
    labels = []
    
    for index, label in csv_file.iterrows():
        filename = label['filename']
        
        # 새로운 파일이면 초기화
        if(filename != prev_filename):
            # 반환할 딕셔너리에 저장
            # 맨 처음 prev_filename == '' 일 때는 
            # files[phase]에 ''가 존재하지 않기 때문에 들어가지 않음
            for phase in ['train', 'val']:
                if prev_filename in files[phase]:
                    file_paths[phase].append(os.path.join(folders[phase], prev_filename))
                    box_coords_and_labels[phase].append(
                        {'boxes': torch.as_tensor(box_coords, dtype=torch.int64),
                         'labels': torch.as_tensor(labels, dtype=torch.int64)}
                    )
                    break
            
            box_coords = []
            labels = []
            prev_filename = filename
        
        # box coord 계산
        region_shape_attributes = eval(label['region_shape_attributes'])
        
        min_x = region_shape_attributes['x']
        min_y = region_shape_attributes['y']
        width = region_shape_attributes['width']
        height = region_shape_attributes['height']
        max_x = int(min_x + width)
        max_y = int(min_y + height)
        
        coord = [min_x, min_y, max_x, max_y]
        
        # 한 이미지의 box coords 모으기
        box_coords.append(coord)
        labels.append(torch.ones(1, dtype=torch.int64))

    # 마지막 항목 저장
    for phase in ['train', 'val']:
        if filename in files[phase]:
            file_paths[phase].append(os.path.join(folders[phase], filename))
            box_coords_and_labels[phase].append(
                {'boxes': torch.as_tensor(box_coords, dtype=torch.int64),
                    'labels': torch.as_tensor(labels, dtype=torch.int64)}
            )
            break

    return file_paths, box_coords_and_labels