# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 12:17:25 2025
"""

import os

#Change project directory
os.chdir(r"PATH TO THE PROJECT FILE")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
from vocoder.bigvgan.models import VocoderBigVGAN
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from wav_evaluation.models.CLAPWrapper import CLAPWrapper
from omegaconf import OmegaConf

import re
from gtts import gTTS
from pydub import AudioSegment
from scipy.io.wavfile import write
import tempfile

import openai

# Constants
SAMPLE_RATE = 16000

# Set device
torch.set_grad_enabled(False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def dur_to_size(duration):
    latent_width = int(duration * 7.8)
    if latent_width % 4 != 0:
        latent_width = (latent_width // 4 + 1) * 4
    return latent_width

def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)["state_dict"], strict=False)
    model = model.to(device)
    model.cond_stage_model.to(model.device)
    model.cond_stage_model.device = model.device
    print(model.device, device, model.cond_stage_model.device)
    sampler = DDIMSampler(model)

    return sampler

def select_best_audio(prompt, wav_list):
    clap_model = CLAPWrapper('useful_ckpts/CLAP/CLAP_weights_2022.pth','useful_ckpts/CLAP/config.yml',use_cuda=torch.cuda.is_available())
    text_embeddings = clap_model.get_text_embeddings([prompt])
    score_list = []
    for data in wav_list:
        sr, wav = data
        audio_embeddings = clap_model.get_audio_embeddings([(torch.FloatTensor(wav), sr)], resample=True)
        score = clap_model.compute_similarity(audio_embeddings, text_embeddings, use_logit_scale=False).squeeze().cpu().numpy()
        score_list.append(score)
    max_index = np.array(score_list).argmax()
    print(score_list, max_index)
    return wav_list[max_index]

def txt2audio(sampler, vocoder, prompt, seed, scale, ddim_steps, n_samples=1, W=624, H=80):
    prng = np.random.RandomState(seed)
    start_code = prng.randn(n_samples, sampler.model.first_stage_model.embed_dim, H // 8, W // 8)
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)
    
    uc = None
    if scale != 1.0:
        uc = sampler.model.get_learned_conditioning(n_samples * [""])
    c = sampler.model.get_learned_conditioning(n_samples * [prompt])
    shape = [sampler.model.first_stage_model.embed_dim, H//8, W//8]  
    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                        conditioning=c,
                                        batch_size=n_samples,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc,
                                        x_T=start_code)

    x_samples_ddim = sampler.model.decode_first_stage(samples_ddim)

    wav_list = []
    for idx, spec in enumerate(x_samples_ddim):
        wav = vocoder.vocode(spec)
        wav_list.append((SAMPLE_RATE, wav))
    best_wav = select_best_audio(prompt, wav_list)
    return best_wav

def generate_audio(prompt, ddim_steps=100, num_samples=3, scale=3.0, seed=44):
    sampler = initialize_model('configs/text_to_audio/txt2audio_args.yaml', 'useful_ckpts/maa1_full.ckpt')
    vocoder = VocoderBigVGAN('vocoder/logs/bigvnat', device=device)
    
    result = txt2audio(
        sampler=sampler,
        vocoder=vocoder,
        prompt=prompt,
        seed=seed,
        scale=scale,
        ddim_steps=ddim_steps,
        n_samples=num_samples,
        H=80, W=624
    )
    
    return result

def save_audio(wav_data, filename):
    import soundfile as sf
    sf.write(filename, wav_data[1], wav_data[0])


# =============================================================================
# #Input Params
# =============================================================================
#prompt = "Heavy breathing" # running 

#Select from audios num.This number control the number of candidates (e.g., generate three audios and choose the best to show you). A Larger value usually lead to better quality with heavier computation
num_samples = 3
#Guidance Scale:(Large => more relevant to text but the quality may drop)
scale=3.0
#Seed:Change this value (any integer number) will lead to a different generation result.
seed=13

# ==============================
# 0. Spatial Audio Narrator
# ==============================

def generate_spatial_audiobook(text):
    """
    Generate ultra-concise spatial audio descriptions matching exact examples,
    inserting sound cues immediately after each sentence.
    
    Parameters:
        text (str): Input text to process
    
    Returns:
        str: Text with single-line sound cues immediately after each sentence
    """
    
    def generate_cues(text_chunk):
        """Generate single-line sound descriptions"""
        prompt = f"""
        Create ONE concise sound description for this text, Similar like the below examples:
        - a cat meowing and young female speaking
        - a group of sheep are baaing
        - a horse galloping
        - Engine noise with other engines passing by
        - a chainsaw cutting as wood cracks and creaks
        - drums and music playing with a man speaking

        Rules:
        1. Only ONE description per batch
        2. Never exceed 10 words
        3. Use simple present tense
        4. No punctuation except commas
        5. Never use numbers or lists
        
        Important: Please avoid noises

        Text: {text_chunk}
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a sound effects generator"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=50  # Strict length limit
        )
        
        raw = response.choices[0].message.content.strip()
        return re.sub(r'[\d\.]+\s*', '', raw)  # Final cleanup

    # --- Processing Logic ---
    sentences = re.split(r'(?<=[.!?])\s+', text)
    annotated = []
    
    for sentence in sentences:
        annotated.append(sentence)
        
        cue = generate_cues(sentence)
        if cue:
            annotated.append(f"[Sound: {cue}]")
    
    return '\n'.join(annotated)

# ==============================
# 1. Configuration
# ==============================
TEMP_DIR = tempfile.TemporaryDirectory()
CONFIG = {
    "crossfade": 150,
    "sfx_duck": -6,
    "headroom": -3,
    "bitrate": "192k",
    "sample_rate": 44100
}

# ==============================
# 2. Sound Generation (Fixed)
# ==============================
def generate_tts_audio(prompt):
    """Generate dummy sine wave audio for demonstration"""
    try:
        sps = CONFIG["sample_rate"]
        freq_hz = 440.0  # Example frequency
        duration_s = 5.0  # Example duration
        
        # Generate sine wave as NumPy array
        each_sample_number = np.arange(duration_s * sps)
        waveform = np.sin(2 * np.pi * each_sample_number * freq_hz / sps)
        waveform_integers = np.int16(waveform * 32767)  # Convert to PCM
        
        # Save as WAV file
        temp_file = os.path.join(TEMP_DIR.name, f"{prompt.replace(' ', '_')}.wav")
        write(temp_file, sps, waveform_integers)
        
        return temp_file
    except Exception as e:
        print(f"SFX Generation Error: {e}")
        return None

def generate_spatial_sfx(description, output_path):
    """Generate spatial sound effect using fixed audio generation"""
    try:
        audio_file = generate_audio(description)
        
        if audio_file:  # Ensure valid file path is returned
            #sfx = AudioSegment.from_file(audio_file)
            #sfx.export(output_path, format="wav")
            save_audio(audio_file, output_path)
            return True
        else:
            raise ValueError("Invalid audio data returned by generate_audio()")
    except Exception as e:
        print(f"SFX Generation Error: {e}")
        return False

# ==============================
# 3. Text Processing and TTS
# ==============================
def parse_content(text):
    text_parts = re.split(r'\[Sound:.*?\]', text)
    text_parts = [part.strip() for part in text_parts if part.strip()]
    
    sound_effects = re.findall(r'\[Sound: (.*?)\]', text)
    
    return list(zip(text_parts, sound_effects + [""]*(len(text_parts)-len(sound_effects))))

def generate_tts(text, output_path):
    try:
        temp_mp3 = os.path.join(TEMP_DIR.name, "temp.mp3")
        tts = gTTS(text=text, lang='en')
        tts.save(temp_mp3)
        
        AudioSegment.from_mp3(temp_mp3).export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"TTS Error: {e}")
        return False

# ==============================
# 4. Mixing Engine and Main Processor
# ==============================
def smart_mix(speech, sfx):
    target_length = max(len(speech), len(sfx))
    speech += AudioSegment.silent(duration=target_length - len(speech))
    sfx += AudioSegment.silent(duration=target_length - len(sfx))
    
    return speech.overlay(sfx.apply_gain(CONFIG["sfx_duck"]))

def create_audiobook(input_text, output_file):
    
    parsed_text = generate_spatial_audiobook(input_text)
    segments = parse_content(parsed_text)
    final_mix = AudioSegment.silent(duration=500)
    
    for idx, (text, sfx_desc) in enumerate(segments):
        print(f"Processing segment {idx+1}/{len(segments)}")
        
        speech_path = os.path.join(TEMP_DIR.name, f"speech_{idx}.wav")
        if not generate_tts(text, speech_path):
            continue
        
        sfx_path = os.path.join(TEMP_DIR.name, f"sfx_{idx}.wav")
        print(sfx_desc)
        print(sfx_path)
        if sfx_desc and not generate_spatial_sfx(sfx_desc, sfx_path):
            sfx_path = None
        
        speech = AudioSegment.from_wav(speech_path)
        sfx = AudioSegment.from_wav(sfx_path) if sfx_path and os.path.exists(sfx_path) else AudioSegment.silent(0)
        
        mixed = smart_mix(speech, sfx)
        final_mix += mixed
    
    final_mix.normalize().apply_gain(CONFIG["headroom"]).export(output_file, format="wav", bitrate=CONFIG["bitrate"])
    
    TEMP_DIR.cleanup()
    print(f"Successfully created {output_file}")

# ==============================
# Example Usage
# ==============================
INPUT_TEXT = """NATHAN RUBIN DIED because he got brave. Not the sustained kind of thing that wins you a medal in a war, but the split-second kind of blurting outrage that gets you killed on the street."""

create_audiobook(INPUT_TEXT, "spatial_audiobook.wav")

# generate_spatial_audiobook(INPUT_TEXT)

# generate_spatial_sfx("girl whistling on road", r"C:\Users\janar\OneDrive\Desktop\student_proj\generate_spatial_audio\girl_whistling.wav")

#---------------------------------------------------------------------------
import re
import os
import tempfile
import numpy as np
from pydub import AudioSegment
from gtts import gTTS
from scipy.io.wavfile import write

# ==============================
# 1. Configuration
# ==============================
TEMP_DIR = tempfile.TemporaryDirectory()
CONFIG = {
    "crossfade": 150,
    "sfx_duck": -6,
    "headroom": -3,
    "bitrate": "192k",
    "sample_rate": 44100
}

# ==============================
# 2. Sound Generation (Fixed)
# ==============================
def generate_tts_audio(prompt):
    """Generate dummy sine wave audio for demonstration"""
    try:
        sps = CONFIG["sample_rate"]
        freq_hz = 440.0  # Example frequency
        duration_s = 5.0  # Example duration
        
        each_sample_number = np.arange(duration_s * sps)
        waveform = np.sin(2 * np.pi * each_sample_number * freq_hz / sps)
        waveform_integers = np.int16(waveform * 32767)  # Convert to PCM
        
        temp_file = os.path.join(TEMP_DIR.name, f"{prompt.replace(' ', '_')}.wav")
        write(temp_file, sps, waveform_integers)
        
        return temp_file
    except Exception as e:
        print(f"SFX Generation Error: {e}")
        return None

def generate_spatial_sfx(description, output_path):
    """Generate spatial sound effect using fixed audio generation"""
    try:
        audio_file = generate_audio(description)  # Your real implementation
        
        if audio_file:
            save_audio(audio_file, output_path)  # Your real implementation
            return True
        else:
            raise ValueError("Invalid audio data returned by generate_audio()")
    except Exception as e:
        print(f"SFX Generation Error: {e}")
        return False

# ==============================
# 3. Text Processing and TTS
# ==============================
def parse_content(text):
    text_parts = re.split(r'\[Sound:.*?\]', text)
    text_parts = [part.strip() for part in text_parts if part.strip()]
    
    sound_effects = re.findall(r'\[Sound: (.*?)\]', text)
    
    return list(zip(text_parts, sound_effects + [""]*(len(text_parts)-len(sound_effects))))

def generate_tts(text, output_path):
    try:
        temp_mp3 = os.path.join(TEMP_DIR.name, "temp.mp3")
        tts = gTTS(text=text, lang='en')
        tts.save(temp_mp3)
        
        AudioSegment.from_mp3(temp_mp3).export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"TTS Error: {e}")
        return False

# ==============================
# 4. Audio Processing Helpers
# ==============================
def normalize_audio(audio_segment, target_dBFS=-20.0):
    """Normalize audio segment to target dBFS"""
    change_in_dBFS = target_dBFS - audio_segment.dBFS
    return audio_segment.apply_gain(change_in_dBFS)

def reduce_noise(audio_segment):
    """
    Placeholder for noise reduction.
    Implement or integrate a noise reduction algorithm here if needed.
    """
    # Example: return audio_segment after processing with noisereduce or other libs
    return audio_segment

# ==============================
# 5. Mixing Engine and Main Processor
# ==============================
def smart_mix(speech, sfx, base_duck=-6):
    # Normalize both audio tracks
    speech = normalize_audio(speech)
    sfx = normalize_audio(sfx)
    
    # Optional noise reduction on SFX
    sfx = reduce_noise(sfx)
    
    # Dynamic ducking: reduce SFX volume relative to speech loudness
    duck_amount = base_duck
    if sfx.dBFS > speech.dBFS:
        duck_amount -= (sfx.dBFS - speech.dBFS)
    sfx = sfx.apply_gain(duck_amount)
    
    # Smooth fade in/out for SFX to avoid abruptness
    sfx = sfx.fade_in(50).fade_out(50)  # 50 ms fades
    
    # Pad shorter audio with silence for equal length
    target_length = max(len(speech), len(sfx))
    speech += AudioSegment.silent(duration=target_length - len(speech))
    sfx += AudioSegment.silent(duration=target_length - len(sfx))
    
    # Overlay SFX on speech
    return speech.overlay(sfx)

def create_audiobook(input_text, output_file):
    
    parsed_text = generate_spatial_audiobook(input_text)
    segments = parse_content(parsed_text)
    final_mix = AudioSegment.silent(duration=500)
    
    for idx, (text, sfx_desc) in enumerate(segments):
        print(f"Processing segment {idx+1}/{len(segments)}")
        
        speech_path = os.path.join(TEMP_DIR.name, f"speech_{idx}.wav")
        if not generate_tts(text, speech_path):
            continue
        
        sfx_path = os.path.join(TEMP_DIR.name, f"sfx_{idx}.wav")
        print(f"SFX description: {sfx_desc}")
        print(f"SFX path: {sfx_path}")
        if sfx_desc and not generate_spatial_sfx(sfx_desc, sfx_path):
            sfx_path = None
        
        speech = AudioSegment.from_wav(speech_path)
        sfx = AudioSegment.from_wav(sfx_path) if sfx_path and os.path.exists(sfx_path) else AudioSegment.silent(0)
        
        mixed = smart_mix(speech, sfx, base_duck=CONFIG["sfx_duck"])
        final_mix += mixed
    
    final_mix = final_mix.normalize().apply_gain(CONFIG["headroom"])
    final_mix.export(output_file, format="wav", bitrate=CONFIG["bitrate"])
    
    TEMP_DIR.cleanup()
    print(f"Successfully created {output_file}")

create_audiobook(INPUT_TEXT, "output/spatial_audiobook.wav")

