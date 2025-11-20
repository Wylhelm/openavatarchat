import json
import os
import numpy as np
from glob import glob


def export_expression_json(flame_param_dir, output_path):
    all_flames_npz = glob(os.path.join(flame_param_dir,'*.npz'))
    all_flames_npz = sorted(all_flames_npz)
    output_json = None
    for flame_path in all_flames_npz:
        flame_param = dict(np.load(flame_path), allow_pickle=True)
        expr = flame_param['expr']
        expr_dict = {f"expr{i}": float(item) for i, item in enumerate(expr.squeeze(0))}
        flame_param['expr'] = expr_dict
        if(output_json is None):
            output_json = flame_param
            try:
                del(output_json['static_offset'])
            except:
                pass
            try:
                del(output_json['allow_pickle'])
            except:
                pass
        else:
            for key in output_json:
                if('shape' in key): continue
                if 'expr' in key:
                    if isinstance(output_json.get(key, None), dict):
                        output_json[key] = [flame_param[key]]
                    else:
                        output_json[key].append(flame_param[key])
                    pass
                else:
                    output_json[key] = np.concatenate([output_json[key], flame_param[key]], axis=0)

    for key in output_json:
        if key != "expr":
            output_json[key] = output_json[key].tolist()

    with open(output_path,'w') as f:
        json.dump(output_json,f)

    print("Save flame params into:", output_path)


if __name__ == '__main__':
    root = './export/nice/'
    # motion_lst = ['qie', 'nice', 'clip1', 'clip2', 'clip3']
    # motion_lst = ['man', 'oldman', 'woman']
    motion_lst = [l.strip() for l in open("./fd_lst.txt").readlines()]
    for motion in motion_lst:
        root = os.path.join('./export/', motion)
        input_dir = os.path.join(root, "flame_param")
        export_expression_json(input_dir, os.path.join(root, 'flame_params.json'))
