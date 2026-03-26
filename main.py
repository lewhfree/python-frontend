import librosa
import ffmpeg
import numpy as np
import glob
from scipy import ndimage
import os
import struct
import xxhash
import json
import requests

NUM_O_FANS = 3
N_FFT=2048
HOP_LENGTH=N_FFT//4
TIME_WINDOW=3
FINAL_SAMPLE_RATE=8000
INBETWEEN_SAMPLE_RATE=11025
DB_CUTOFF_VAL=-38
PLOT_SEARCH_SIZE=15
FRAME_WIDTH_T=HOP_LENGTH/FINAL_SAMPLE_RATE
SERVER_HOSTNAME="http://localhost"
SERVER_PORT=8080

def genDataPairs(filename):
    stream=ffmpeg.input(filename)
    stream=ffmpeg.output(stream, "audio.tmp.mp3", ar=INBETWEEN_SAMPLE_RATE, ac=1)
    stream.run(overwrite_output=True, quiet=True)
    stream=ffmpeg.input("audio.tmp.mp3")
    stream=ffmpeg.output(stream, "audio.out.mp3", ar=FINAL_SAMPLE_RATE, ac=1)
    stream.run(overwrite_output=True, quiet=True)
    
    audio, sr = librosa.load("audio.out.mp3", mono=True, sr=FINAL_SAMPLE_RATE)
    
    stft = librosa.stft(y=audio, hop_length=HOP_LENGTH, n_fft=N_FFT)
    stft = np.abs(stft)
    stft = librosa.amplitude_to_db(stft, ref=np.max)
    
    betterPlot = ndimage.maximum_filter(stft, size=PLOT_SEARCH_SIZE)
    mask = (stft == betterPlot) & (stft > DB_CUTOFF_VAL)
    ret = np.full_like(stft, -80.0)
    ret[mask] = stft[mask] 
   
    f_indices, t_indices = np.where(mask)
    result = list(zip(t_indices, f_indices))
    formatted_tuples = [(int(f), int(t)) for f, t in result]
    sorted_result = sorted(formatted_tuples, key=lambda x: (x[0], x[1]))#list is touples of time, freq

    os.remove("audio.tmp.mp3")
    os.remove("audio.out.mp3")
    
    return sorted_result

def computeAndSend(fileName, apiEndpoint, spotifyKey=None):
    file = fileName
    hashes = []
    times = []
    results = genDataPairs(file)
    for i in range(len(results)):
        origin = results[i]
        window_min = TIME_WINDOW
        fans_found = 0
        search_limit = min(100, len(results) - i)
        for j in range(1, search_limit):
            if ((origin[0] + window_min) <= results[i+j][0]) and fans_found < NUM_O_FANS:
                window_min += 1
                fans_found += 1 
                point = results[i+j]
                delta_t = point[0] - origin[0]
                origin_freq = origin[1]
                end_freq = point[1]
                packed_bytes = struct.pack("<III", delta_t, origin_freq, end_freq)
                hash = xxhash.xxh64(packed_bytes).hexdigest()
                t=int(FRAME_WIDTH_T*origin[0]*1000)
                times.append(t)
                hashes.append(hash)

    if spotifyKey!=None:
        json_object = {
            "spotify": spotifyKey,
            "hashlist": hashes,
            "offsetlist": times,
        }
    else:    
        json_object = {
            "hashlist": hashes,
            "offsetlist": times,
        }
    x=requests.post(SERVER_HOSTNAME+":"+str(SERVER_PORT)+"/"+apiEndpoint, json=json_object)
    print(x.text)

    print("uploaded file")

computeAndSend("./party.mp3", "multiget")
