# import this
# pip install pipwin
# pipwin install pyaudio
# enable Stereo Mix in Audio Devices - Recording
# install ffmpeg from binary, add into path, reload IDE


import wave
import pyaudio
import subprocess
from pynput import keyboard
import os


def on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
        if key.char == 'q':
            # Stop listener
            return False
    except AttributeError:
        print('special key {0} pressed'.format(
            key))


chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 2
fs = 44100  # Record at 44100 samples per second
seconds = 300
filename = "output4"

p = pyaudio.PyAudio()  # Create an interface to PortAudio

for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if 'Stereo Mix' in dev['name']:
        dev_index = dev['index']
        print('dev_index', dev_index)
        break

print('Recording')

stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                input_device_index=dev_index,
                frames_per_buffer=chunk,
                input=True)

frames = []  # Initialize array to store frames

# Store data in chunks for 3 seconds
listener = keyboard.Listener(on_press=on_press)
listener.start()
while True:
    if not listener.running:
        break
    data = stream.read(chunk)
    frames.append(data)

# Stop and close the stream
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print('Finished recording')

# Save the recorded data as a WAV file
wf = wave.open(f'{filename}.wav', 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()

# merge_into_movie = f'ffmpeg -y -i {filename}.avi -i {filename}.wav -c copy {filename}.mkv'
# p = subprocess.Popen(merge_into_movie)
# output, _ = p.communicate()
# print(output)
# os.remove(f'{filename}.avi')
# os.remove(f'{filename}.wav')
