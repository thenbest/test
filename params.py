dataset_name='train/mid_quality'

feat_name = 'log_mel'
# change to 3 for 3-class classification
num_classes=3
# For binary classification only
#num_classes=1
class_labels = ['good', 'fair', 'bad']
num_feats=128
num_mels=128
sr=22050
num_frames, max_dur=[2580, 1*60]

ws=2048
hs=512
fft=2048
train_list_path= r"/data/home/v_rxwtang/trans_torch/csv_path/"
train_feats_path = r"/data/home/v_rxwtang/trans_torch/rmc_mel_10/"
# test_list_path ="test_list/test_speed.csv"
# test_feats_path = "test_docker/feat_b66fa069acb60fe0a86ebc29927ca12f/"
use_transform = False
