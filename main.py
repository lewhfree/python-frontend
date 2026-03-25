import librosa
import ffmpeg
import numpy as np
import glob
from scipy import ndimage
import os

def genDataPairs(filename, searchSize, cutoff):
    plotSearchSize = searchSize
    DBCutoffVal = cutoff

    stream=ffmpeg.input(filename)
    stream=ffmpeg.output(stream, "audio.tmp.mp3", ar=11025, ac=1)
    stream.run(overwrite_output=True, quiet=True)
    stream=ffmpeg.input("audio.tmp.mp3")
    stream=ffmpeg.output(stream, "audio.out.mp3", ar=8000, ac=1)
    stream.run(overwrite_output=True, quiet=True)
    
    audio, sr = librosa.load("audio.out.mp3", mono=True, sr=8000)
    
    stft = librosa.stft(y=audio)
    stft = np.abs(stft)
    stft = librosa.amplitude_to_db(stft, ref=np.max)
    
    betterPlot = ndimage.maximum_filter(stft, size=plotSearchSize)
    mask = (stft == betterPlot) & (stft > DBCutoffVal)
    ret = np.full_like(stft, -80.0)
    ret[mask] = stft[mask] 
   
    f_indices, t_indices = np.where(mask)
    time_in_seconds = librosa.frames_to_time(t_indices, sr=sr)
    freq_map = librosa.fft_frequencies(sr=sr, n_fft=2048)
    freq_in_hz = freq_map[f_indices]
    result = list(zip(freq_in_hz, time_in_seconds))
    pps = len(result) // librosa.get_duration(y=audio, sr=sr)
    formatted_tuples = [(round(float(f), 2), round(float(t), 2)) for f, t in result]
    sorted_result = sorted(formatted_tuples, key=lambda x: x[0])
    print(sorted_result)

    os.remove("audio.tmp.mp3")
    os.remove("audio.out.mp3")
    
    return pps, len(result)

files = glob.glob("music/*")

ppss = []
totals = []
for file in files:
    pps, total = genDataPairs(file, 15, -38)
    ppss.append(pps)
    totals.append(total)
