import numpy as np
import librosa.display
from sklearn.mixture import GaussianMixture

# ---------------------------- Read me ---------------------------
# "filename"부분에 테스트파일명을 입력해주세요
# 결과는 console창에 출력됩니다
test_file = "C:/Users/user0425/Desktop/DigitalSound/filename.wav"
# ----------------------------------------------------------------


path = [
    "C:/Users/user0425/Desktop/DigitalSound/F1tr.wav",
    "C:/Users/user0425/Desktop/DigitalSound/F2tr.wav",
    "C:/Users/user0425/Desktop/DigitalSound/F3tr.wav",
    "C:/Users/user0425/Desktop/DigitalSound/M1tr.wav",
    "C:/Users/user0425/Desktop/DigitalSound/M2tr.wav",
    "C:/Users/user0425/Desktop/DigitalSound/M3tr.wav"
]

name = [
    "F1's voice",
    "F2's voice",
    "F3's voice",
    "M1's voice",
    "M2's voice",
    "M3's voice"
]

gmm = []

# ------------------- Create GMM ---------------------
for i in range(6):
    file = path[i]
    signal, sample_rate = librosa.load(file, sr=16000)
    mfccs = librosa.feature.mfcc(
        y=signal, sr=16000, n_mfcc=24, n_mels=24, n_fft = 512,hop_length = 512)
    mfccs = np.transpose(mfccs)
    gmm.append(GaussianMixture(n_components=6, covariance_type='full', max_iter=100, init_params='kmeans'))
    gmm[i].fit(mfccs)



# --------------- Create Test mfccs -----------------

test_signal, sample_rate = librosa.load(test_file, sr=16000)
test_mfccs = librosa.feature.mfcc(y=test_signal, sr=16000, n_mfcc=24, n_mels=24, n_fft = 512, hop_length = 512)
test_mfccs = np.transpose(test_mfccs)


# ---------------- Decision ----------------

# float 0~1 -> log : -999999 ~ 0
# score가 1에 가까울수록 log값은 0에 수렴함
min_scr = abs(gmm[0].score(test_mfccs))
res = name[0]
for i in range(6):
    scr = abs(gmm[i].score(test_mfccs))
    if min_scr > scr:
        min_scr = scr
        res = name[i]

print(res)