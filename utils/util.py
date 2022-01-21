import os
import pickle
import json
import numpy as np


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)
    return paths


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def clear_dir(path):
    import shutil
    if os.path.exists(path):
        shutil.rmtree(path)
    return mkdir(path)


def load_pickle_file(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


def write_pickle_file(pkl_path, data_dict):
    with open(pkl_path, 'wb') as fp:
        pickle.dump(data_dict, fp, protocol=2)


def read_json(fpath):
    with open(fpath,'r') as f:
        obj=json.load(f)
    return obj

def write_json(fpath, data_dict):
    with open(fpath, 'w') as f:
        json.dump(data_dict, f, indent=4)
        
def load_obj(obj_file):
    with open(obj_file, 'r') as fp:
        verts = []
        faces = []
        vts = []
        vns = []
        faces_vts = []
        faces_vns = []

        for line in fp:
            line = line.rstrip()
            line_splits = line.split()
            prefix = line_splits[0]

            if prefix == 'v':
                verts.append(np.array([line_splits[1], line_splits[2], line_splits[3]], dtype=np.float32))

#             elif prefix == 'vn':
#                 vns.append(np.array([line_splits[1], line_splits[2], line_splits[3]], dtype=np.float32))

            elif prefix == 'vt':
                vts.append(np.array([line_splits[1], line_splits[2]], dtype=np.float32))

            elif prefix == 'f':
                f = []
                f_vt = []
#                 f_vn = []
                for p_str in line_splits[1:4]:
                    p_split = p_str.split('/')
                    f.append(p_split[0])
                    # f_vt.append(p_split[1])
#                     f_vn.append(p_split[2])

                faces.append(np.array(f, dtype=np.int32) - 1)
                # faces_vts.append(np.array(f_vt, dtype=np.int32) - 1)
#                 faces_vns.append(np.array(f_vn, dtype=np.int32) - 1)

            else:
#                 raise ValueError(prefix)
                continue

        obj_dict = {
            'vertices': np.array(verts, dtype=np.float32),
            'faces': np.array(faces, dtype=np.int32),
            # 'vts': np.array(vts, dtype=np.float32),
#             'vns': np.array(vns, dtype=np.float32),
            # 'faces_vts': np.array(faces_vts, dtype=np.int32),
#             'faces_vns': np.array(faces_vns, dtype=np.int32)
        }

        return obj_dict

def save_to_obj(verts, faces, path):
    """
    Save the SMPL model into .obj file.
    Parameter:
    ---------
    path: Path to save.
    """

    with open(path, 'w') as fp:
        fp.write('g\n')
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
        fp.write('s off\n')