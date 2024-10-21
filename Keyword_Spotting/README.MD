This program is designed to create a quantized binary model for deployment on an embedded system (such as Arduino). 
If you are recording your own audio files with your voice for a custom target word, your .ogg files will be converted to .wav at 16kHz.

Here a link to the instructions on how to run locally a speech recording application:
https://github.com/tinyMLx/appendix/blob/main/OpenSpeechRecordingLocal.md

Make sure to download the audio data that will be necessary for any KWS application:
https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz

You can find the extract louder section program compiled in this directory or download it here:
https://github.com/petewarden/extract_loudest_section.git

Parts of the instructions I got from this project are from the HarvardX TinyML3/Deploying TinyML. 
They provided a Google collab to create the binary model for the embedded system, this is my python adaptation of it for local application.

To finally run this on arduino, you need to download on Arduino IDE a library with a KWS example.
Such as: Harvard_TinyMLx by TinyMLx Authors
