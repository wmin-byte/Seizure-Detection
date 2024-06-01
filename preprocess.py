import pyedflib
import pywt
import numpy as np
from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

i_dict = {
    1: 42,
    2: 36,
    3: 38,
    4: 42,
    5: 39,
    6: 18,
    7: 19,
    8: 20,
    9: 19,
    10: 25,
    11: 35,
    12: 24,
    13: 33,
    14: 26,
    15: 40,
    16: 19,
    17: 21,
    18: 36,
    19: 30,
    20: 29,
    21: 33,
    22: 31,
    23: 9
}

lowcut = 0.5
highcut = 50.0

for chbi in range(1, 24):
    j_max = i_dict.get(chbi) + 1
    for chbj in range(1, j_max):
        edf_file = pyedflib.EdfReader('CHB-MIT/chb' + "{}".format(chbi) + '/' + "{}".format(chbj) + '.edf')
        num_channels = edf_file.signals_in_file
        signal_labels = edf_file.getSignalLabels()
        num_channels = []
        desired_labels = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
                          "FZ-CZ", "CZ-PZ", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8",
                          "T8-P8", "P8-O2"]
        found_T8_P8 = []
        for label in desired_labels:
            if label in signal_labels and label not in found_T8_P8:
                chanel_index = (signal_labels.index(label))
                num_channels.append(chanel_index)
                if label == "T8-P8":
                    found_T8_P8.append(label)
        selected_channels = num_channels

        coeffs_list = []

        sample_frequency = edf_file.getSampleFrequency(0)


        previous_num_samples = 0

        for channel in selected_channels:

            data = edf_file.readSignal(channel)


            filtered_data = butter_bandpass_filter(data, lowcut, highcut, sample_frequency)


            wavelet = 'db4'
            level = 5
            coeffs = pywt.wavedec(filtered_data, wavelet, level=level)

            coeffs_list.append(coeffs)

            num_samples = len(filtered_data)

            time = np.arange(1 + previous_num_samples, num_samples + previous_num_samples + 1) / sample_frequency

            previous_num_samples += num_samples

        edf_file.close()

        output_file = 'DWT/chb' + "{}".format(chbi) + '/' + "{}".format(chbj) + '.txt'
        with open(output_file, 'w') as f:
            np.set_printoptions(threshold=np.inf)
            for channel in range(len(selected_channels)):
                coeffs = coeffs_list[channel]
                k = 0
                for i, coeff in enumerate(coeffs):
                    for j, value in enumerate(coeff):
                        k = k + 1
                        f.write(f'{k}, {channel + 1}, {value},\n')
        print("saved chb:", chbi, "-", chbj, "files")
