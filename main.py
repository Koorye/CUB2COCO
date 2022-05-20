import json
import os
import pandas as pd


root = 'D:/Datasets/CUB_200_2011'

def load_txt(path, dtype=None):
    path = os.path.join(root, path)
    with open(path) as f:
        lines = f.readlines()

    contents = []
    for line in lines:
        line = line.strip()
        content = line.split(' ')[1:]

        if dtype is not None:
            for i in range(len(content)):
                content[i] = dtype(content[i])
        
        if len(content) == 1:
            content = content[0]
        contents.append(content)

    return contents

imgs = load_txt('images.txt')
labels = load_txt('image_class_labels.txt', dtype=int)
labels = list(map(lambda x: x - 1, labels))
bboxes = load_txt('bounding_boxes.txt', dtype=float)
train_test_flags = load_txt('train_test_split.txt', dtype=int)
content_df = pd.DataFrame({
    'file_name': imgs,
    'category_id': labels,
    'bbox': bboxes,
    'is_train': train_test_flags,
})
content_df['area'] = content_df['bbox'].apply(lambda x: x[2] * x[3])

cls_df = pd.read_csv(os.path.join(root, 'classes.txt'), sep=' ', header=None)
cls_df.columns = ['id', 'name']
cls_df['id'] -= 1
cats = cls_df.to_dict('records')

train_df = content_df[content_df['is_train']==1].drop(columns='is_train')
test_df = content_df[content_df['is_train']==0].drop(columns='is_train')
train_df = train_df.reset_index(drop=True).reset_index().rename(columns={'index': 'id'})
test_df = test_df.reset_index(drop=True).reset_index().rename(columns={'index': 'id'})
train_df['image_id'] = train_df['id']
test_df['image_id'] = test_df['id']

imgs_train = train_df[['id', 'file_name']].to_dict('records')
imgs_test = test_df[['id', 'file_name']].to_dict('records')

anns_train = train_df[['id', 'image_id', 'category_id', 'bbox', 'area']].to_dict('records')
anns_test = test_df[['id', 'image_id', 'category_id', 'bbox', 'area']].to_dict('records')

with open('instances_train.json', 'w') as f:
    json.dump(dict(
        categories=cats,
        annotations=anns_train,
        images=imgs_train,
    ), f, indent=2)
with open('instances_test.json', 'w') as f:
    json.dump(dict(
        categories=cats,
        annotations=anns_test,
        images=imgs_test,
    ), f, indent=2)

