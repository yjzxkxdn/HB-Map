import numpy as np
import torch
import librosa
import soundfile as sf
from torchfcpe import spawn_bundled_infer_model


from HBMap import HBMap


def test_split_audio_by_harmonics():

    audio_path = "女-雨下一整晚.wav"
    device = "cpu"
    hop_size = 512

    model = spawn_bundled_infer_model(device=device)
    audio, sr = librosa.load(audio_path, sr=None)
    
    # Pad audio
    pad_len = hop_size - (len(audio) % hop_size)
    if pad_len != hop_size:
        audio = np.pad(audio, (0, pad_len))
    audio_length = len(audio)

    f0_tensor = model.infer(
        torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(-1),
        sr=sr,
        decoder_mode='local_argmax',  # Recommended mode
        threshold=0.006,  # Threshold for V/UV decision
        f0_min=80,  # Minimum pitch
        f0_max=880,  # Maximum pitch
        interp_uv=True,  # Interpolate unvoiced frames
        output_interp_target_length=audio_length//hop_size,  # Interpolate to target length
    )  
    src_f0 = f0_tensor.squeeze().numpy()

    HBM = HBMap(hop_size, sr)
    sub_audios = HBM.split_audio_by_harmonics(src_f0, audio)

    for i, sub_audio in enumerate(sub_audios):
        sf.write(f"harmonic_bands_stft2/sub_audio_{i}.wav", sub_audio, sr)

    sum_audio = np.sum(sub_audios, axis=0)
    sf.write("sum_audio.wav", sum_audio, sr)


def test_shift_f0(new_f0=None):

    audio_path = "女-雨下一整晚.wav"
    device = "cpu"
    hop_size = 512

    model = spawn_bundled_infer_model(device=device)
    audio, sr = librosa.load(audio_path, sr=None)
    
    # Pad audio
    pad_len = hop_size - (len(audio) % hop_size)
    if pad_len != hop_size:
        audio = np.pad(audio, (0, pad_len))
    audio_length = len(audio)

    f0_tensor = model.infer(
        torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(-1),
        sr=sr,
        decoder_mode='local_argmax',  # Recommended mode
        threshold=0.006,  # Threshold for V/UV decision
        f0_min=80,  # Minimum pitch
        f0_max=880,  # Maximum pitch
        interp_uv=True,  # Interpolate unvoiced frames
        output_interp_target_length=audio_length//hop_size,  # Interpolate to target length
    )  
    src_f0 = f0_tensor.squeeze().numpy()

    HBM = HBMap(hop_size, sr)
    sub_audios = HBM.STFT_base_split_audio_to_harmonics_band(src_f0, audio)
    
    if new_f0 is None:
        new_f0 = src_f0 * 2**(-5/12)
        '''audiof0, srf0 = librosa.load("Untitled2f0.wav", sr=None)
        # Pad audio
        pad_len = hop_length - ( len(audiof0) % hop_length)
        if pad_len != hop_length:
            audiof0 = np.pad(audiof0, (0, pad_len))
        f0_nwe_tensor = model.infer(
            torch.from_numpy(audiof0).float().unsqueeze(0).unsqueeze(-1),
            sr=sr,
            decoder_mode='local_argmax',  # Recommended mode
            threshold=0.006,  # Threshold for V/UV decision
            f0_min=80,  # Minimum pitch
            f0_max=880,  # Maximum pitch
            interp_uv=True,  # Interpolate unvoiced frames
            output_interp_target_length=audio_length//hop_length,  # Interpolate to target length
        )  
        new_f0 = f0_nwe_tensor.squeeze().numpy()'''



    final_audio = HBM.HBMap_shift_f0(new_f0, src_f0, sub_audios)
    
    sf.write("女-雨下一整晚 - Vocal低-5.wav", final_audio, sr)
    print("Done.")


if __name__ == "__main__":
    import time
    start = time.time()
    # test_split_audio_by_harmonics()
    test_shift_f0()
    print(time.time() - start)