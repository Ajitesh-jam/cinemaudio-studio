import soundfile as sf
from IPython.display import Audio
from tango2.tango import Tango

tango = Tango("declare-lab/tango2")

# prompt = "Piercing, terrified girl's scream"
prompts=["wail and moan", "Zipper clothing"]
print("starting generation")
audios = tango.generate_for_batch(prompts,disable_progress=False)
print("generation complete")
for i, audio in enumerate(audios):
    sf.write(f"{prompts[i]}.wav", audio, samplerate=16000)
    Audio(data=audio, rate=16000)