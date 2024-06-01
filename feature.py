import numpy as np
from scipy.stats import skew, kurtosis
import os

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

for chbi in range(12, 24):
    j_max = i_dict.get(chbi) + 1
    for chbj in range(1, j_max):
        input_file = 'DWT/chb' + "{}".format(chbi) + '/' + "{}".format(chbj) + '.txt'
        if os.path.getsize(input_file) == 0:
            print("Empty file. Skipping:", chbi, "-", chbj)
            continue
        with open(input_file, 'r') as f:
            lines = f.readlines()

        data_list = []
        for line in lines:
            line = line.strip()
            if line:
                line_data = line.split(", ")
                line_data[2] = line_data[2].replace(",", "")
                data_list.append(line_data)

        # 获取通道数量
        num_channels = max([int(data[1]) for data in data_list])

        statistics = {}

        segment_length = 1
        sample_frequency = 256
        num_samples = segment_length * sample_frequency
        num_segments = len(data_list) // num_samples

        for channel in range(1, num_channels + 1):
            channel_data = [float(data[2]) for data in data_list if int(data[1]) == channel]

            for i in range(num_segments):
                segment_start = i * num_samples
                segment_end = (i + 1) * num_samples
                segment_values = channel_data[segment_start:segment_end]


                if len(segment_values) < num_samples:
                    continue

                segment_max = np.max(segment_values)
                segment_min = np.min(segment_values)
                segment_median = np.median(segment_values)
                segment_mean = np.mean(segment_values)
                segment_std = np.std(segment_values)
                segment_variance = np.var(segment_values)
                segment_mad = np.mean(np.abs(segment_values - segment_mean))
                segment_rms = np.sqrt(np.mean(np.square(segment_values)))
                segment_skewness = skew(segment_values)
                segment_kurtosis = kurtosis(segment_values)

                if channel not in statistics:
                    statistics[channel] = []

                statistics[channel].append({
                    "Segment": i + 1,
                    "Max": segment_max,
                    "Min": segment_min,
                    "Median": segment_median,
                    "Mean": segment_mean,
                    "Standard Deviation": segment_std,
                    "Variance": segment_variance,
                    "Mean Absolute Deviation": segment_mad,
                    "Root Mean Square": segment_rms,
                    "Skewness": segment_skewness,
                    "Kurtosis": segment_kurtosis,
                    "Label": 0
                })

        output_file = 'Statistics3/chb' + "{}".format(chbi) + '/' + "{}".format(chbj) + '.csv'
        with open(output_file, 'w') as f:
            f.write(
                "Channel,Segment,Max,Min,Median,Mean,Standard Deviation,Variance,Mean Absolute Deviation,Root Mean Square,Skewness,Kurtosis,Label\n")

            for channel in statistics:
                for data in statistics[channel]:
                    f.write(
                        f"{channel},{data['Segment']},{data['Max']},{data['Min']},{data['Median']},{data['Mean']},{data['Standard Deviation']},{data['Variance']},{data['Mean Absolute Deviation']},{data['Root Mean Square']},{data['Skewness']},{data['Kurtosis']},{data['Label']}\n")

        print("saved：", chbi, "-", chbj)
