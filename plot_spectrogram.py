import torchaudio
import matplotlib.pyplot as plt
import requests

# Test music
fileName = 'test.wav'
url = "https://pytorch.org/tutorials//_static/img/steam-train-whistle-daniel_simon-converted-from-mp3.wav"
r = requests.get(url)

# Save
with open(fileName, 'wb') as f:
    f.write(r.content)

waveform, sample_rate = torchaudio.load(fileName)

print('Shape of waveform: {}'.format(waveform.size()))
print('Sample rate of waveform: {}'.format(sample_rate))

plt.figure()
plt.plot(waveform.t().numpy())
plt.show()

# Spectrogram
specgram = torchaudio.transforms.Spectrogram()(waveform)
print('Shape of spectrogram:', specgram.size())

plt.figure()
plt.imshow(specgram.log2()[0, :, :].numpy())
plt.show()