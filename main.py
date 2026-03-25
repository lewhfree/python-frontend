from future.moves.sys import set_coroutine_origin_tracking_depth
import librosa
import math
import ffmpeg
import numpy as np
import matplotlib
import scipy
import xxhash
from scipy import ndimage

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
filename: str = "audio.flac"

stream=ffmpeg.input(filename)
stream=ffmpeg.output(stream, "audio.out.mp3", ar=11025, ac=1)
stream.run(overwrite_output=True, quiet=True)

audio, sr = librosa.load("audio.out.mp3", mono=True, sr=11025)

stft = librosa.stft(y=audio)
stft = np.abs(stft)
stft = librosa.amplitude_to_db(stft, ref=np.max)
# print(stft)
# print(stft.shape)

hash = xxhash.xxh64("test").hexdigest()
print(hash)
betterPlot = ndimage.maximum_filter(stft, size=8)
mask = (stft == betterPlot) & (stft > -70)
ret = np.full_like(stft, -80.0)  # start with -80 everywhere
ret[mask] = stft[mask] 
# librosa.display.specshow(ret, sr=sr, x_axis='time', y_axis='hz')
# plt.ylim(0, 5100)
# plt.colorbar(format='%+2.0f dB')
# plt.title('STFT Magnitude (dB)')
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (Hz)")
# plt.tight_layout()
# plt.show()

#constelation.
#
result = []
num_freqs, num_times = ret.shape

for t in range(num_times):
    for f in range(num_freqs):
        if ret[f, t] >= -70:  # Ignore values that are -80 (silence)
            result.append((f, t))  # Append a tuple (frequency, time)

# print(result)
print(len(result))

sorted_result = sorted(result, key=lambda x: x[1])

# print(sorted_result)
