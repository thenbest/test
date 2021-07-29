import librosa
import numpy as np
import easy_vad
import matplotlib.pyplot as plt
import librosa.display
import os
from ezSpleeter.audio.adapter import get_default_audio_adapter
from ezSpleeter.separator import Separator
from easy_vad.pyvad.effects import split
from scipy.io.wavfile import write
sample_rate = 44100
def save_audio(data, fs=22050,name="test"):
    scaled = np.int16(data / np.max(np.abs(data)) * 32767)
    write(name + '.mp3', fs, scaled)

SR=22050
N_FFT=2048
HOP_LENGTH=512
N_MELS=128
OFFSET=0
DURATION=120
MEL='log_mel'
SPLEETER_TYPE = '2stems'
VAD_MODE=3

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
class Preprocessor:
    '''
    spleeter -> vad -> mel_spectrum
    '''
    def __init__(self):
        self.separator = Separator(SPLEETER_TYPE,stft_backend='tensorflow')
        self.adapter = get_default_audio_adapter()
        self.split = split

    def load(self, fn):
        source, _ = self.adapter.load(fn, offset=OFFSET, duration=DURATION, sample_rate=SR)
        return source

    def get_vocals(self, input):
        output = self.separator.separate(input,SPLEETER_TYPE)
        vocals = output['vocals']           # (T, 2)
        vocals = vocals.T                   # (2, T)
        vocals = librosa.to_mono(vocals)    # (T, )

        return vocals

    def vad(self, input):

        edges = self.split(input, SR, vad_mode=VAD_MODE)
        y = []
        for edge in edges:
            y.append(input[edge[0]:edge[1]])
        if y == []:
            #print("Error in preprocesser.py is catched.")
            return y
        y = np.concatenate(y)

        return y

    def extract_feature(self, input, mel):

        feat = librosa.feature.melspectrogram(y=input, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)

        if mel == 'log_mel':
            return np.log10(1 + 10 * feat)
        else:
            return feat

    def __call__(self, fn):
        y = self.load(fn)
        y = self.get_vocals(y)
        y = self.vad(y)
        if y == []:
            return y
        # save_audio(y, fs=22050,name="test")
        # y = self.extract_feature(y, MEL)

        return y
if __name__ == '__main__':
    base_path = '/data_share2/v_chujiewu/rmc_data/ugc_m4a_file/good_m4a/'
    tar_dir = '/data/home/v_rxwtang/trans_torch/rmc_data/good'
    if not (os.path.isdir(tar_dir)):
        os.mkdir(tar_dir)
    filenames = os.listdir(base_path)
    lst_files = [ filename for filename in filenames if filename.endswith( '.m4a' ) ]
    size = len(lst_files)
    preprocessor = Preprocessor()
    file_num = len(lst_files)
    print(file_num)
    for index, file in enumerate(lst_files):
        try:
            if index % 10 == 0:
                print("finish:" + str(index) + "  " + str(round(index/file_num,2)))
            file_path= os.path.join(base_path, file)
            feat = preprocessor(file_path)
            if len(feat) > 2425500: # 110s*22050hz
                print("pass")
                continue
            save_audio(feat, fs=22050,name=os.path.join(tar_dir, file[:-4]))
            np.save(os.path.join(tar_dir, file[:-4])+'.npy',feat)
            size = size - 1
        except:
            print("something is wrong")

        
        
        
        
        
        
        
        
        
        
        
        
        
        