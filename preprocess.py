import os
from glob import glob

from absl import app, flags
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import pandas as pd
from librosa.core import resample
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string('src_root', 'wave_files', 'Directory of audio files in total duration.')
flags.DEFINE_string('dst_root', 'preprocessed', 'Directory to put audio files split by delta_time.')
flags.DEFINE_string('file_name', '3a3d0279', 'File to plot over time to check magnitude.')
flags.DEFINE_integer('threshold', 20, 'Threshold magnitude for np.int16 dtype')
flags.DEFINE_float('delta_time', 1.0, 'Time in seconds to sample audio.')
flags.DEFINE_integer('sample_rate', 16000, 'Rate to down sample audio.')


def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)


def down_sample(path, sr):
    rate, wav = wavfile.read(path)
    wav = resample(wav.astype(np.float32), rate, sr)
    wav = wav.astype(np.int16)
    return sr, wav


def get_audio_envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate / 20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean


def test_audio_envelop():
    src_root = FLAGS.src_root
    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_path = [x for x in wav_paths if FLAGS.file_name in x]
    if len(wav_path) != 1:
        print('audio file not found for sub-string: {}'.format(FLAGS.file_name))
        return
    rate, wav = down_sample(wav_path[0], FLAGS.sample_rate)
    mask, env = get_audio_envelope(wav, rate, threshold=FLAGS.threshold)
    plt.style.use('ggplot')
    plt.title('Signal Envelope, Threshold = {}'.format(str(FLAGS.threshold)))
    plt.plot(wav[np.logical_not(mask)], color='r', label='remove')
    plt.plot(wav[mask], color='c', label='keep')
    plt.plot(env, color='m', label='envelope')
    plt.grid(False)
    plt.legend(loc='best')
    plt.show()


def save_fragmented_wav(sample, rate, target_dir, file_name, index):
    file_name = file_name.split('.wav')[0]
    dst_path = os.path.join(target_dir.split('.')[0], file_name + '_{}.wav'.format(str(index)))
    if os.path.exists(dst_path):
        return
    wavfile.write(dst_path, rate, sample)


def split_wav_files():
    src_root = FLAGS.src_root
    dst_root = FLAGS.dst_root
    dt = FLAGS.delta_time

    check_dir(dst_root)
    classes = os.listdir(src_root)
    for _cls in classes:
        target_dir = os.path.join(dst_root, _cls)
        check_dir(target_dir)
        src_dir = os.path.join(src_root, _cls)
        for fn in tqdm(os.listdir(src_dir)):
            src_fn = os.path.join(src_dir, fn)
            rate, wav = down_sample(src_fn, FLAGS.sample_rate)
            mask, y_mean = get_audio_envelope(wav, rate, threshold=FLAGS.threshold)
            wav = wav[mask]
            delta_sample = int(dt * rate)

            # cleaned audio is less than a single sample
            # pad with zeros to delta_sample size
            if wav.shape[0] < delta_sample:
                sample = np.zeros(shape=(delta_sample,), dtype=np.int16)
                sample[:wav.shape[0]] = wav
                save_fragmented_wav(sample, rate, target_dir, fn, 0)

            # step through audio and save every delta_sample
            # discard the ending audio if it is too short
            else:
                trunc = wav.shape[0] % delta_sample
                for cnt, i in enumerate(np.arange(0, wav.shape[0] - trunc, delta_sample)):
                    start = int(i)
                    stop = int(i + delta_sample)
                    sample = wav[start:stop]
                    save_fragmented_wav(sample, rate, target_dir, fn, cnt)


def main(argv):
    # test_audio_envelop()
    split_wav_files()


if __name__ == '__main__':
    app.run(main)
