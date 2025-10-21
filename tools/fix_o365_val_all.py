# Written by Mingxuan Liu

import argparse
import json
import copy
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", default='datasets/objects365/zhiyuan_objv2_val.json')
    parser.add_argument("--fix_name_map", default='datasets/metadata/Objects365_names_fix.csv')
    parser.add_argument("--img_dir", default='datasets/objects365/val/')
    args = parser.parse_args()

    # Fix names
    new_names = {}
    old_names = {}
    with open(args.fix_name_map, 'r') as f:
        for line in f:
            tmp = line.strip().split(',')
            old_names[int(tmp[0])] = tmp[1]
            new_names[int(tmp[0])] = tmp[2]

    data = json.load(open(args.ann, 'r'))

    cat_info = copy.deepcopy(data['categories'])
    
    for x in cat_info:
        if old_names[x['id']].strip() != x['name'].strip():
            print('{} {} {}'.format(x, old_names[x['id']], new_names[x['id']]))
            import pdb; pdb.set_trace()
        if old_names[x['id']] != new_names[x['id']]:
            print('Renaming', x['id'], x['name'], new_names[x['id']])
            x['name'] = new_names[x['id']]
    
    data['categories'] = cat_info

    # Fix Missing
    images = []
    count = 0
    for x in data['images']:
        path = '{}/{}'.format(args.img_dir, x['file_name'])
        if os.path.exists(path):
            images.append(x)
        else:
            print(path)
            count = count + 1
    print('Missing', count, 'images')
    data['images'] = images
    out_name = args.ann[:-5] + '_fixed_everything.json'
    # print('Saving to', out_name)
    # json.dump(data, open(out_name, 'w'))


