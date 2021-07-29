import os
from multiprocessing import Pool
from os import path as osp
from tempfile import NamedTemporaryFile
import warnings

warnings.filterwarnings('ignore')

import librosa
import numpy as np

from .pyvad import split


def do_vad_on_wave(wave, sr=22050):
    edges = split(wave, sr, vad_mode=3)
    y = []
    for edge in edges:
        y.append(wave[edge[0]:edge[1]])
    y = np.concatenate(y)
    return y


def do_vad(input_path, output_path):
    if osp.exists(output_path):
        print(output_path, "is already exist.")
        return
    if not osp.exists(input_path):
        print(input_path, "does not exist.")
        return
    cvt_mp3_cmd = 'ffmpeg -i "{in_file}" -f mp3 -ab 128k -y "{out_file}"'
    x, sr = librosa.load(input_path, res_type="polyphase")
    edges = split(x, sr, vad_mode=3)
    y = []
    for edge in edges:
        y.append(x[edge[0]:edge[1]])
    y = np.concatenate(y)
    output_root = os.path.split(output_path)[0]
    if output_root != "" and not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)
    
    with NamedTemporaryFile(suffix=".wav") as f:
        tmp_file = f.name
        librosa.output.write_wav(tmp_file, y, sr)
        os.system(cvt_mp3_cmd.format(in_file=tmp_file, out_file=output_path))
    print("output:", output_path)


def do_vad_on_music(input_vocal, input_music, output_path):
    if os.path.exists(output_path):
        print(output_path, "is already exist.")
        return
    if not osp.exists(input_vocal):
        print(input_vocal, "does not exist.")
        return
    if not osp.exists(input_music):
        print(input_music, "does not exist.")
        return
    cvt_mp3_cmd = 'ffmpeg -i "{in_file}" -f mp3 -ab 128k -y "{out_file}"'
    vocal, sr = librosa.load(input_vocal, res_type="polyphase")
    music, _ = librosa.load(input_music, res_type="polyphase", sr=sr)
    edges = split(vocal, sr, vad_mode=3)
    y = []
    if vocal.shape[0] != music.shape[0]:
        print("length error", input_music)
        return
    for edge in edges:
        y.append(music[edge[0]:edge[1]])
    y = np.concatenate(y)
    output_root = os.path.split(output_path)[0]
    if output_root != "" and not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)
    with NamedTemporaryFile(suffix=".wav") as f:
        tmp_file = f.name
        librosa.output.write_wav(tmp_file, y, sr)
        cmd = cvt_mp3_cmd.format(in_file=tmp_file, out_file=output_path)
        os.system(cvt_mp3_cmd.format(in_file=tmp_file, out_file=output_path))
    print("output:", output_path)


def do_vad_on_music_multi(input_vocals, input_musics, output_paths, n_threads=8):
    pool = Pool(n_threads)
    for input_vocal, input_music, output_path in zip(input_vocals, input_musics, output_paths):
        pool.apply_async(do_vad_on_music, args=(input_vocal, input_music, output_path))
    pool.close()
    pool.join()


def do_vad_multi(input_paths, output_paths, n_threads=8):
    pool = Pool(n_threads)
    for input_path, output_path in zip(input_paths, output_paths):
        pool.apply_async(do_vad, args=(input_path, output_path))
    pool.close()
    pool.join()
