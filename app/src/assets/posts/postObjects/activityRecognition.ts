import Post from "../postModel";

export default new Post(
  // title
  "Activity Recognition with Smartphone Sensors",
  // subtitle
  "Using signal processing and machine learning to find out what you're doing",
  // publishDate
  new Date("2022-02-05"),
  // titleImageUrl
  "https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/activity-recognition/activitiyTitle.jpg",
  // titleImageDescription
  "Can your smartphone's sensors tell you what you're doing?",
  // tags
  ["Data Science & AI/ML", "Learning"],
  // content
  `**TL;DR:** I used sensor data from smartphones' accelerometers and gyroscopes for human activity recognition by extracting features from their Fourier transform spectrum, power spectral density, and auto-correlation function, and training a XGBoost classifier on these features.

Besides computer vision tasks, I may have some work coming up soon which involves working with data from a smartphone's sensors, such as the accelerometer and gyroscope. I haven't worked much in the realm of digital signal processing since I left academic research 3 years ago, so I though I should probably refresh my skills in that field, and of course combine it with machine learning. I wasn't to keen on collecting a large data set "just for fun" by myself so I looked for some accelerometer/gyroscope data sets online for training a model and then being able to do predictions with my own phone. I don't have a car myself so that's out for me, and I haven't found a data set for phones on bicycles, so I guess I'll have to do some moves with my own body. I found a promising data set for human activity recognition online, which I'll describe in more detail below. Here, I'll do a proof of concept to see if I can extract features from the data to train a machine learning model with. I'd like to later build a progressive web app with predictive functionality, but unfortunately the numpy/scipy ecosystem isn't really available in JavaScript (though there are some partial implementations), and I'll need to see if it's not too much of a hassle to implement this myself. Anyways, this POC is already exiting, so let's have a look at the data.

### The Data Set

The data set I use is the [Smartphone-Based Recognition of Human Activities and Postural Transitions Data Set](http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions), that is, an updated version of the original data set that also contains the raw recorded data. I downloaded it from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php) (donated by Jorge L. Reyes-Ortiz et al., you can find links to their papers on the data set site), but you can also find it on [Kaggle](https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones) with plenty of code examples. This is a well know data set, and building a well-performing classifier is not much of a challenge. The authors already provide 561 pre-engineered features that have been derived from the raw data. The sensor signals have been pre-processed with noise filters and then sampled in fixed-width sliding windows of 2.56 seconds and 50% overlap, meaning 128 readings/window at a sampling frequency of 50 Hz. From each of these windows, the authors calculated for each sensor and axis the derivative (they called it jerk) and magnitude over all axes. On all XYZ axes, their jerks, and magnitudes, they then calculated various quantities from both the time and the frequency domain such as mean, standard deviation, median absolute deviation, max and min values, signal magnitude area, energy, interquartile range, signal entropy, correlation between two signals, mean frequencies, skewness, kurtosis, and some more (for details, please refer to their repo's dataset description in the file \`features_info.txt\`). This adds up to the proud number of 561 features, with which one can easily train a XGBoost classifier that scores 0.93 accuracy and F1.

If this is already a well solved problem, why do I even care about this data set? Well, because the authors already kinda took the whole fun out of it by doing all the feature engineering. I want to learn about signal processing, I want to play with raw sensor data, calculate their spectra, and look what the peaks in there tell me. So, I'll choose a different approach as compared to the authors of the data set. Instead of computing quantities over the whole sampled windows in the time and frequency domain, I'll start from the raw data, filter out gravity and noise, calculate the Fourier transform, power spectral density (using Welch's method), and auto-correlation function for each window, and use their peaks as features to train a machine learning model on. I'll also try to make the data somewhat orientation-independent by transforming them with a principal component analysis (PCA) and see how the results compare to the data with the original orientation.

Let's start by reading the data from file and see how the initial data frame looks like:

\`\`\`python
from typing import Tuple, List, Dict, Union
import pandas as pd
import numpy as np
import scipy
from glob import glob

# load the labels from file
labels = pd.read_csv(
    "./HAPT Data Set/RawData/labels.txt",
    header=None,
    sep=" ",
    names=["experiment", "user", "activity", "start", "stop"]
)

# load the raw sensor data from file
df = pd.DataFrame([], columns=[
    "acc_x", "acc_y", "acc_z", "acc_total",
    "gyro_x", "gyro_y", "gyro_z", "gyro_total",
    "activity"
])
# by looping over the data files of 61 experiments
for i in range(1, 62):
    acc_df = pd.read_csv(
        glob(f"./HAPT Data Set/RawData/acc_exp{str(i).zfill(2)}_*.txt")[0],
        header=None,
        sep=" ",
        names=["acc_x", "acc_y", "acc_z"]
    )
    acc_df["acc_total"] = np.sqrt(
        acc_df["acc_x"]**2 + acc_df["acc_y"]**2 + acc_df["acc_z"]**2
    )

    gyro_df = pd.read_csv(
        glob(f"./HAPT Data Set/RawData/gyro_exp{str(i).zfill(2)}_*.txt")[0],
        header=None,
        sep=" ",
        names=["gyro_x", "gyro_y", "gyro_z"]
    )
    gyro_df["gyro_total"] = np.sqrt(
        gyro_df["gyro_x"]**2 + gyro_df["gyro_y"]**2 + gyro_df["gyro_z"]**2
    )

    user = glob(f"./HAPT Data Set/RawData/gyro_exp{str(i).zfill(2)}_*.txt")[0][-6:-4]
    df_merged = acc_df.merge(gyro_df, left_index=True, right_index=True)
    df_merged["experiment"] = i
    df_merged["user"] = int(user)
    df_merged["activity"] = np.NaN

    # fill in the labels
    for _, label in labels[labels["experiment"] == i].iterrows():
        df_merged["activity"].iloc[label["start"]:label["stop"]+1] = label["activity"]

    df = pd.concat([df, df_merged], axis=0)

# get rid of the activity transitions, they're not of interest
df = df[df.activity.between(1,6)]

df.sample(5)
\`\`\`

\`\`\`python
          acc_x     acc_y     acc_z  acc_total    gyro_x    gyro_y    gyro_z  gyro_total  activity  experiment  user
5696   0.944444 -0.029167 -0.326389   0.999678 -0.024740  0.029627 -0.005192    0.038946       4.0        29.0  14.0
17287  0.543056 -0.043056  0.031944   0.545695 -0.789543  0.349415 -0.208610    0.888250       3.0         5.0   3.0
15789  1.119445 -0.506944 -0.081944   1.231610  0.099571  0.905913 -0.113926    0.918462       2.0        42.0  21.0
12060  0.956944 -0.008333 -0.294444   1.001254  0.146302  0.066279 -0.180205    0.241394       5.0        34.0  17.0
5166   0.279167  0.445833  0.848611   0.998420  0.019548  0.003054  0.000916    0.019806       6.0        32.0  16.0
\`\`\`

As you can see in the \`activity\` columns, there are activities numbered 1 to 6 to be classified:

\`\`\`
1 WALKING
2 WALKING_UPSTAIRS
3 WALKING_DOWNSTAIRS
4 SITTING
5 STANDING
6 LAYING
\`\`\`

Let's have a look how the raw sensor data looks like for these activities. I'll just show you the plots here and not the full \`matplotlib\` code, as it is pretty verbose and doesn't really add anything to the understanding.

![Sensor data corresponding to the different activities.](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/activity-recognition/activities.svg)

## Preprocessing the data

There are two things that I want to get rid off before building features: gravity and noise. The accelerometer measures the total acceleration, which means it also measures gravity. The positioning of the smartphones in the authors' experiments is always constant so gravity should always appear along the same axis (and obviously stay constant), so it wouldn't add any useful information here. I've read [in a paper](https://arxiv.org/pdf/1805.03368.pdf) of some method of using the gravity contribution on all three axes to transform an arbitrarily oriented sensor coordinate system to one that is uniformly oriented, thereby making the data collection orientation-independent, but that's not what I'll do here. The authors' of the data set separated the gravity acceleration from the data with a filter, and that's what I'll to as well. I'll assume that the gravity contribution to the acceleration is of very low frequency, below 0.3 Hz, and use a high-pass filter to only let frequencies higher than 0.3 pass. Furthermore, most sensors will collect some noise in addition to the actual signal. This noise is usually of high frequencies, at least compared to normal human movements. The sampling frequency of these measurements is 50 Hz, so no frequency higher than half of that (the Nyquist frequency of 25 Hz) could be captured anyway. I'll assume that a humans shouldn't be able to do any movement more than 15 times a second, so I'll also apply a low-pass filter with a cutoff frequency of 15 Hz.

I'll use a Butterworth filter as it's designed to have a frequency response as flat as possible in the passband, which means it doesn't alter the part of the signal that we want to keep too much. \`Scipy.signal\` offers diverse functionality for designing signal filters and applying them. We'll do this in two steps here: Getting the Butterworth filter coefficients and than apply the filter. The function used to apply the filter here is \`scipy.signal.filtfilt\`, which applies a linear digital filter twice, once forward and once backwards. This results in no phase shift in the signal, which probably isn't relevant for this use case, but could e.g. be important when denoising a spectrogram where the exact peak position really matters.

\`\`\`python
from scipy.signal import butter, filtfilt, welch
from scipy.fft import fft, fftfreq

# function to design the filter coefficients
def butter_coef(
    cutoff: float,
    fs: float,
    order: int = 5,
    btype: str = "low"
) -> Tuple[np.array, np.array]:
    nyq = fs / 2
    normalized_cutoff = cutoff / nyq
    b, a = butter(order, normalized_cutoff, btype=btype, analog=False)
    return b, a

# function applying the filter
def butter_filter(
    data: np.array,
    cutoff: float,
    fs: float,
    order: int = 5,
    btype: str = "low"
) -> np.array:
    b, a = butter_coef(cutoff, fs, order=order, btype=btype)
    y = filtfilt(b, a, data, method="gust")
    return y
\`\`\`

Now, let's apply the Butterworth high-pass filter to separate out gravitational acceleration from the acceleration signals and a low-pass filter for noise reduction to all raw signals. Note that no gravitational contribution has to be removed from the gyroscope data; that's the reason why I don't combine both filters into one band-pass filter.

\`\`\`python
# set some parameters first
cutoff_lp=15       # cutoff frequency (in Hz) for the low-pass filter
cutoff_hp=0.3      # cutoff frequency (in Hz) for the high-pass filter
order=5            # order of the Butterworth filter
fs = 50.0          # data sampling frequency (in Hz)

# apply the filters
for col in df.columns.values[:-3]:
    # filter out gravity
    if col.startswith("acc"):
        df[col] = butter_filter(
            df[col],
            cutoff=cutoff_hp,
            fs=fs, order=order,
            btype="high"
        )

    # filter out noise
    df[col] = butter_filter(
        df[col],
        cutoff=cutoff_lp,
        fs=fs,
        order=order,
        btype="low"
    )
\`\`\`

Let's have a look at how the data looks before and after filtering:

![Sensor data before and after filtering.](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/activity-recognition/filtered.svg)

### Orientation-independence

As I mentioned before, I thought it may be a good idea to think about orientation-independence when trying to recognize human activities from smartphone sensor data. In the experiments from which the data set originates, the smartphones where always mounted in the same way with the same orientation at the hips of the users. This is nice if you want to investigate in which direction an activity accelerates the phone, relative to the user. But in reality, a user might have the phone crammed into any of her pockets or backpack or where ever, and would certainly not care about keeping the phone in the same orientation relative to their bodies at all times. There are a couple of ways we could do that. For example we could rotate the coordinate system in a way that maximizes the gravitational contribution along one axis an say that's the vertical axis. We could then consider the magnitude of the two remaining axis as one horizontal component, which is done in [this paper](https://arxiv.org/pdf/1805.03368.pdf). But I kinda want to keep three axes and I already removed gravity, so let's do something else. Think about variability: If you would plot every sensor measurement in 3D you'll end up with a blob of data points that may be oriented in different ways depending on the orientation of the phone, but should have a characteristic shape, depending on the measured activity. The blobs should then have a direction in which the variance is the highest. So, if we do a principal component analysis and find 3 principal components, we should be able to preserve the whole variance of the data but find a new coordinate system (the 3 ortho-normal principal components along the 3 directions of highest variance in the data) into which we can transform the data. I found [a paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5579846/) doing basically the same using the first 3 singular values and corresponding matrix columns from a singular value decomposition.

How exactly does that work? We consider that for every sampled window of data, in principal the orientation could be different, so we'll do the transform for each window of 2.56 seconds individually, based on the XYZ components of accelerometer and gyroscope. Since the two types of sensors should have a constant orientation relative to each other, we'll concat the data from both sensors and find the common principal components. To bring the variance of the two sensors onto the same scale, we'll standard-scale them first. Having the 3 principal components, we transform the data into the new coordinate system and separate the measurements of the two sensors again:

\`\`\`python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def orientation_invariant_transform(
    df_acc_xyz: pd.DataFrame,
    df_gyro_xyz: pd.DataFrame
) -> pd.DataFrame:
    # scale the data by removing the mean and scaling to unit variance
    df_acc_xyz = pd.DataFrame(
        StandardScaler().fit_transform(df_acc_xyz),
        columns=df_acc_xyz.columns
    )
    df_gyro_xyz = pd.DataFrame(
        StandardScaler().fit_transform(df_gyro_xyz),
        columns=df_gyro_xyz.columns
    )

    # concat data from both sensors
    concat = np.concatenate([df_acc_xyz.values, df_gyro_xyz.values])

    # transform data into coordinate system of the principal components
    concat_oi = PCA(n_components=3).fit_transform(concat)

    # separate the two sensors' data again
    df_acc_oi = pd.DataFrame(
        concat_oi[:len(concat_oi)//2],
        columns=df_acc_xyz.columns
    )
    df_gyro_oi = pd.DataFrame(
        concat_oi[len(concat_oi)//2:],
        columns=df_gyro_xyz.columns
    )
    # calculate the magnitude / Euclidean norm
    df_acc_oi["acc_total"] = np.sqrt(
        df_acc_oi["acc_x"]**2
        + df_acc_oi["acc_y"]**2
        + df_acc_oi["acc_z"]**2
    )
    df_gyro_oi["gyro_total"] = np.sqrt(
        df_gyro_oi["gyro_x"]**2
        + df_gyro_oi["gyro_y"]**2
        + df_gyro_oi["gyro_z"]**2
    )

    return df_acc_oi.merge(df_gyro_oi, left_index=True, right_index=True)
\`\`\`

I only define the function here, it will be called later when the features for each data window are generated.

### Features from time-frequency domain transformations

I've explained in some detail the features that the authors of the data set have generated from both the time and the frequency domain. I'd like to create features here of which the meaning can be understood in a conceptual and graphical way. Specifically, I want to apply certain transforms to the data from which I will then extract the position and height of peaks. The transforms/functions I'll use are the Fourier transform, the power spectral density (using Welch's method), and the auto-correlation function.

Let's quickly explain what these three are. Essentially, they all give us information about the periodicities in the signal. Imagine your smartphone recording your acceleration while you walk. I've you're walking at a constant speed, the bumps in the acceleration signal should be spaced evenly over time. Maybe there will be some quick jerk movements that you do while walking that may appear in the signal at shorter intervals than your main steps. These movements overlay, making up the resulting signal, similar to different musical notes combining into one resulting sound wave. Just as with sound waves, we can either look at how the amplitude of the wave changes over time, or we transform the signal into the frequency domain and see which frequencies contribute with which amplitude to the periodic signal. If we'd do that with the walking signal, we may see a large contribution at a lower frequency, corresponding to the main steps, and maybe some smaller contribution at higher frequencies, corresponding to quicker jerk movements. The Fourier transform (FT) does exactly that transformation from the time domain (where we see the raw signal) into the frequency domain (where we see the spectrum of the signal), by decomposing the original signal into its contributing frequencies. The power spectral density (PSD) is conceptually very similar. It calculates how the power of the signal is distributed over the frequencies in its spectrum, i.e. it's the spectral energy distribution that would be found per unit time. To calculate the PSD, we use [Welch's method](https://en.wikipedia.org/wiki/Welch%27s_method), which computes an estimate of the power spectral density by dividing the data into overlapping segments, computing a modified power spectrum for each segment and averaging the spectra. Lastly, the auto-correlation function kinda stays in the time domain but also measures a the periodicity of a signal. It calculates the correlation of a signal with a delayed copy of itself as a function of delay/time-lag. Basically that means, if a periodic sine signal exhibits a maximum every 2 seconds, the autocorrelation function would also show a peak at 2, 4, 6, 8 ... and so on seconds because that's where the lagged version of the signal is similar (or identical, in case of a noiseless signal) as the un-lagged signal.

This will all make much more sense if we plot it. In the plot below, we can see the original (already filtered) signal from the accelerometer when walking downstairs, its spectrum (the frequency domain representation) created by the Fourier transform, the power spectral density, and the autocorrelation function.

![Original accelerometer data, its Fourier transform, power spectral density, and autocorrelation function.](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/activity-recognition/features.svg)

You can see that the main "humps" in the original signal are about 0.65 seconds apart. This seems to be how long it takes to take a step when walking downstairs. We can find this main interval as well in the autocorrelation function where we find peaks at multiples of ca. 0.65 seconds of lag and in the Fourier transform and power spectral density, where we can see that the largest contribution to the spectrum and its power comes from a corresponding base frequency of ca. 1.54 Hz. I've also marked some of the most prominent peaks occurring in these plots. The position and intensity of these peaks is what I'll use as features for the machine learning model here.

I'm not aware of a function that readily implements the estimated autocorrelation function, but we can easily calculate it using \`numpy.correlate\` to correlate the input with itself. We'll also normalize it by subtracting the mean and dividing by the variance:

\`\`\`python
def estimated_autocorrelation(x: np.array) -> np.array:
    N = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-N:]
    result = r/(variance*(np.arange(N, 0, -1)))
    return result
\`\`\`

Scipy provides functionality for calculating the the fourier transform (\`scipy.fft.fft\`) and its corresponding frequencies (\`scipy.fft.fftfreq\`), as well as Welch's method (\`scipy.signal.welch\`). The Fourier transform returns an array of complex numbers and for a real signal, it should be symmetric around 0, so we'll do a couple of things here: Take the magnitude of the complex numbers (with \`abs\`), drop all frequencies below 0, and normalize the outputs. So, for each component (meaning each axis and the magnitude of each sensor), we'll now get the 5 highest peaks of the Fourier transform, the power spectral density, and the autocorrelation function:

\`\`\`python
# we'll consider only the top 5 highest peaks
top_n_peaks = 5

def find_n_highest_peaks(
    x: np.array,
    y: np.array,
    top_n: int
) -> Tuple[np.array, np.array]:
    # find all present peaks
    peaks = scipy.signal.find_peaks(y)[0]
    heights = y[peaks]

    # if there are enough peaks, sort them from decreasing height
    if len(peaks) >= top_n:
        idx = heights.argsort()[:-top_n-1:-1]
        peaks = peaks[idx]
        return x[peaks], y[peaks]

    # if there are not enough peaks, sort the present ones
    # from decreasing height and pad the rest with zeros
    else:
        n_missing_peaks = top_n - len(peaks)
        idx = heights.argsort()[::-1]
        peaks = peaks[idx]
        return (
            np.concatenate([x[peaks], np.zeros(n_missing_peaks)]),
            np.concatenate([y[peaks], np.zeros(n_missing_peaks)])
        )

def get_peak_features_for_component(
    signal: pd.DataFrame,
    component: str,
    top_n_peaks: int,
    fs: float
) -> pd.DataFrame:
    N = len(signal)

    # FFT - Fast Fourier Transform
    y_fft = 2.0/N * np.abs(fft(signal[component].values)[:N//2])
    x_fft = fftfreq(N, 1.0 / fs)[:N//2]
    peaks_fft_x, peaks_fft_y = find_n_highest_peaks(x_fft, y_fft, top_n_peaks)
    peaks_fft_sum = peaks_fft_x + peaks_fft_y

    # PSD - Power Spectral Denisty using Welch's method
    x_psd, y_psd = welch(signal[component].values, fs=fs)
    peaks_psd_x, peaks_psd_y = find_n_highest_peaks(x_psd, y_psd, top_n_peaks)
    peaks_psd_sum = peaks_psd_x + peaks_psd_y

    # ACF - estimated auto-correlation function
    y_acf = estimated_autocorrelation(signal[component].values)
    x_acf = np.array([1/fs * n for n in range(0, N)])
    peaks_acf_x, peaks_acf_y = find_n_highest_peaks(x_acf, y_acf, top_n_peaks)
    peaks_acf_sum = peaks_acf_x + peaks_acf_y

    # create the column names for the features of this component
    # per component there are
    # 3 (fft, psd, acf) * 3 (x, y, sum) * top_n_peaks features
    columns = []
    for feat in [
        "peaks_fft_x", "peaks_fft_y", "peaks_fft_sum",
        "peaks_psd_x", "peaks_psd_y", "peaks_psd_sum",
        "peaks_acf_x", "peaks_acf_y", "peaks_acf_sum",
    ]:
        for i in range(top_n_peaks):
            columns.append(f"{component}_{feat}_{i}")

    feature_values = np.concatenate([
        peaks_fft_x, peaks_fft_y, peaks_fft_sum,
        peaks_psd_x, peaks_psd_y, peaks_psd_sum,
        peaks_acf_x, peaks_acf_y, peaks_acf_sum
    ]).reshape(1,-1)

    return pd.DataFrame(feature_values, columns=columns)
\`\`\`

This will get us all the features we need for one component (meaning each axis and the magnitude of each sensor) and one time window. To process the entire data set of raw data, we'll loop over all time windows and components. Just as the authors of the data set did, we'll use fixed-width sliding windows of 2.56 seconds and 50% overlap, meaning 128 readings/window at a sampling frequency of 50 Hz:

\`\`\`python
# the data is sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window)
window_length = 128
overlap_length = window_length // 2

def create_features(
    df: pd.DataFrame,
    window_length: int,
    overlap_length: int,
    top_n_peaks: int,
    fs:float,
    oi_transform: bool,
    calc_jerk: bool = True
) -> pd.DataFrame:
    # the final output will be saved here
    df_features = None

    # loop over all windows
    for i in range(0, len(df), overlap_length):
        signal = df.iloc[i:i+window_length]

        # if the window contains more than 1 activity, skip it
        if signal["activity"].nunique() > 1:
            continue

        # if desired, to the orientation-invariant transform
        if oi_transform:
            df_oi = orientation_invariant_transform(
                signal[["acc_x", "acc_y", "acc_z"]],
                signal[["gyro_x", "gyro_y", "gyro_z"]]
            )
            signal = pd.concat(
                [df_oi, signal.iloc[:,-3:].reset_index(drop=True)],
                axis=1
            )

        # loop over all components and create features for them
        feature_row = None
        for component in signal.columns[:-3]:
            component_features = get_peak_features_for_component(
                signal,
                component,
                top_n_peaks,
                fs
            )
            feature_row = (feature_row.merge(
                               component_features,
                               left_index=True,
                               right_index=True
                           )
                           if not feature_row is None
                           else component_features)

        # set the label for the window
        feature_row["activity"] = signal["activity"].iloc[0]

        # append the window to the final feature data frame
        df_features = (pd.concat([df_features, feature_row])
                       if not df_features is None
                       else feature_row)

    return df_features.reset_index(drop=True)
\`\`\`

### Training a machine learning model

Now we finally have everything together to see how a model would perform on these features. The authors of the data set have provided a train-test-split that ensures that no users are in both the training and test set. This way we test if a model learns to truly generalize well from one person to another and not just learns the moves of a particular person by hard.

\`\`\`python
with open("./HAPT Data Set/Train/subject_id_train.txt", "r") as file:
    train_subjects = set([float(u) for u in file.read().splitlines()])

with open("./HAPT Data Set/Test/subject_id_test.txt", "r") as file:
    test_subjects = set([float(u) for u in file.read().splitlines()])

print(
    f"Test fraction: " \
    f"{len(test_subjects) / (len(train_subjects) + len(test_subjects))}"
)
\`\`\`

\`\`\`python
Test fraction: 0.3
\`\`\`

Let's create the feature data for both the data with the original orientation as well as for the data transformed to be orientation-independent:

\`\`\`python
# for the original orientation
# train set
df_features_train = create_features(
    df[df["user"].isin(train_subjects)],
    window_length,
    overlap_length,
    top_n_peaks,
    fs,
    oi_transform=False
)
# test set
df_features_test = create_features(
    df[df["user"].isin(test_subjects)],
    window_length,
    overlap_length,
    top_n_peaks,
    fs,
    oi_transform=False,
)
# split features and labels
X_train = df_features_train.drop("activity", axis=1)
y_train = df_features_train["activity"]
X_test = df_features_test.drop("activity", axis=1)
y_test = df_features_test["activity"]

# for the orientation-independent transformation
# train set
df_features_oi_train = create_features(
    df[df["user"].isin(train_subjects)],
    window_length,
    overlap_length,
    top_n_peaks,
    fs,
    oi_transform=True
)
# test set
df_features_oi_test = create_features(
    df[df["user"].isin(test_subjects)],
    window_length,
    overlap_length,
    top_n_peaks,
    fs,
    oi_transform=True,
)
# split features and labels
X_train_oi = df_features_oi_train.drop("activity", axis=1)
y_train_oi = df_features_oi_train["activity"]
X_test_oi = df_features_oi_test.drop("activity", axis=1)
y_test_oi = df_features_oi_test["activity"]
\`\`\`

I've ran a small randomized hyper-parameter optimization and came up with the following hyper-parameters for a XGBoost classifier:

\`\`\`python
best_params = {
'n_estimators': 300,
 'min_child_weight': 1,
 'max_depth': 3,
 'lambda': 1,
 'eta': 0.5
}
\`\`\`

Now let's see how well the two versions of features perform:

\`\`\`python
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

labels = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING"
]

# train and score a classifier on the original orientation data
clf = XGBClassifier(**best_params).fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test), target_names=labels))
\`\`\`

\`\`\`
                    precision    recall  f1-score   support

           WALKING       0.98      0.99      0.98       514
  WALKING_UPSTAIRS       0.94      0.97      0.96       449
WALKING_DOWNSTAIRS       0.98      0.94      0.96       402
           SITTING       0.82      0.82      0.82       483
          STANDING       0.86      0.90      0.88       540
            LAYING       0.92      0.88      0.90       530

          accuracy                           0.91      2918
         macro avg       0.92      0.91      0.92      2918
      weighted avg       0.91      0.91      0.91      2918
\`\`\`

\`\`\`python
# train and score a classifier on the orientation-independent data
clf_oi = XGBClassifier(**best_params).fit(X_train_oi, y_train_oi)
print(classification_report(
          y_test_oi,
          clf_oi.predict(X_test_oi),
          target_names=labels
))
\`\`\`

\`\`\`
                    precision    recall  f1-score   support

           WALKING       0.96      0.93      0.94       514
  WALKING_UPSTAIRS       0.95      0.94      0.94       449
WALKING_DOWNSTAIRS       0.92      0.93      0.93       402
           SITTING       0.51      0.47      0.49       483
          STANDING       0.66      0.77      0.71       540
            LAYING       0.69      0.65      0.66       530

          accuracy                           0.77      2918
         macro avg       0.78      0.78      0.78      2918
      weighted avg       0.77      0.77      0.77      2918
\`\`\`

We can see that the original model performs pretty well, at an overall accuracy of 91%, it's only 2% worse than what I scored with the features provided by the authors of the data set. It is evident that it performs particularly well at recognizing the activities that involve a lot of movement such as walking (up or downstairs), but performs worse on the "still" activities of sitting, standing, and laying. This is even more clearly visible on the orientation-independent data. The overall accuracy has dropped a lot to 77% percent, but if we look more closely at the individual classes we see that this is mostly to a decrease in performance for the "still" activities. The walking activities are still recognized fairly well. This makes sense if we remember how we try to get the orientation-independence: We transform the data into a new coordinate basis defined by 3 principal components. These are the ortho-normal components along the directions of highest variance in the data. For the walking activities, where we have a lot of movement, the variance along certain directions should be clearly higher than along other directions; hence, the transformation works relatively reliably for different examples. But for the "still" activities, the movements and the recorded acceleration/angular velocity probably results only from body shaking or breathing, is much smaller in magnitude and variance, and the variance is probably less directional. Hence, the transformation works less reliably for different examples. I think using the principal component analysis is a good option for activities that involve sufficient amounts of movement, but I'd like to apply it to data from many sensors that have actually been mounted in different orientation to see if it really works. Something like that was done with singular value composition in [this paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5579846/), and their results are pretty good.

### Conclusion

We've used a data set containing sensor data from smartphones for recognizing human activities. We have seen how we can use digital signal processing techniques to filter noise from the raw data, calculate various transforms on it such as the Fourier transform, power spectral density, and autocorrelation function, and extract their peaks as features for a machine learning model. We've also seen how principal component analysis can be used to transform the data in a way that is invariant to sensor orientation. Finally, we've trained XGBoost classifiers on both versions (original orientation and orientation-independent) of the data. Both versions of features perform well on activities that involve a lot of movement, but the one transformed with principal component analysis has trouble recognizing activities with less movement due to the low variance in the corresponding data.
`
);
