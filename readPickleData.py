import pickle

with open('data/mosi_data_noalign.pkl', 'rb') as handle:
    mosi_data = pickle.load(handle, encoding='latin1')

print(mosi_data)
# audio_train = mosi_data['audio_train']
# audio_test = mosi_data['audio_test']
# video_train = mosi_data['video_train']
# video_test = mosi_data['video_test']
# text_train = mosi_data['text_train']
# text_test = mosi_data['text_test']
# train_mask = mosi_data['train_mask']
# test_mask = mosi_data['test_mask']
# train_label = mosi_data['train_label']
# test_label = mosi_data['test_label']