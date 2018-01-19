import os
import numpy as np
from scipy import signal
from scipy.io import wavfile


class data_loader(object):

    def __init__(self, dir_):

        music_files = [f for f in os.listdir(dir_) if ".wav" in f]

        music_data = []
        for f in music_files:
            fs, data = wavfile.read(os.path.join(dir_, f))
            data_downsampled = data_loader.resample(fs, data, 8000)
            music_data.append(data_downsampled)

        self.concatenate_music(music_data)


    def concatenate_music(self, music_data):

        music_in = np.concatenate(music_data)
        music_out = music_in[1:]
        music_in = music_in[:-1]
        
        split = int(0.3*len(music_in))

        self.music_in_train = music_in[:split]
        self.music_in_test = music_in[split:]
        self.music_out_train = music_out[:split]
        self.music_out_test = music_out[split:]


    def get_random_train_batch(self, batch_size, sequence_length):

        choices = np.random.choice(len(self.music_in_train) - sequence_length - 1,
                                   batch_size)

        batch_in_list = [self.music_in_train[i:i+sequence_length]
                         for i in choices]
        batch_out_list = [self.music_out_train[i:i+sequence_length]
                          for i in choices]

        batch_in = np.stack(batch_in_list, axis=0)
        batch_out = np.stack(batch_out_list, axis=0)

        return batch_in, batch_out


    def get_random_test_batch(self, batch_size, sequence_length):

        choices = np.random.choice(len(self.music_in_test) - sequence_length - 1,
                                   batch_size)

        batch_in_list = [self.music_in_test[i:i+sequence_length]
                         for i in choices]
        batch_out_list = [self.music_out_test[i:i+sequence_length]
                          for i in choices]

        batch_in = np.stack(batch_in_list, axis=0)
        batch_out = np.stack(batch_out_list, axis=0)
        return batch_in, batch_out


    @staticmethod
    def resample(fs_t, data, fs_d):

        secs = int(len(data)/fs_t) # True sampling frequency
        samples = secs*fs_d # Desired sampling frequency
        data_resampled = np.expand_dims(signal.resample(data, int(samples))[:, 0], axis=-1)
        return data_resampled


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    dl = data_loader("music")
    bi, bo = dl.get_random_train_batch(10, 500)

    plt.figure()
    for i in range(10):
        plt.plot(bi[i])
        plt.plot(bo[i])
    plt.show()