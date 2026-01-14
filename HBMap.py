import numpy as np
from scipy.signal import stft, istft, hilbert

from utils import find_segments, get_interpolated_mask

class HBMap:
    def __init__(self, hop_size, sr, f0_min=30, f0_max=2000):
        self.hop_size = hop_size
        self.sr = sr
        self.f0_min = f0_min
        self.f0_max = f0_max

    def STFT_base_split_audio_to_harmonics_band(self, f0_frames, audio, f0_quantile=0.1):
        """
        按 f0 谐波中点分割 STFT
        
        Args:
            f0_frames: (n_frames,) array-like of f0 (Hz)
            audio: (n_samples,) 1D array
        
        Returns:
            sub_audios: list of (n_samples,) arrays
        """
        n_samples = len(audio)
        win_length = self.hop_size * 2
        n_fft = win_length
        f_nyq = self.sr / 2.0

        f0 = np.asarray(f0_frames, dtype=np.float64)
        n_frames = len(f0)

        # 1. STFT
        f, t, Zxx = stft(
            audio,
            fs=self.sr,
            window='hann',
            nperseg=win_length,
            noverlap=win_length - self.hop_size,
            nfft=n_fft,
            return_onesided=True,
            boundary=None
        )  # Zxx shape: [n_freq, n_frames_stft]
        
        n_freq, n_frames_stft = Zxx.shape
        freqs = f  # already in Hz, shape [n_freq]

        if n_frames > n_frames_stft:
            f0 = f0[:n_frames_stft]
        elif n_frames < n_frames_stft:
            f0_pad = np.full(n_frames_stft - n_frames, -1.0, dtype=np.float64)
            f0 = np.concatenate([f0, f0_pad])
        # Now f0: [n_frames_stft]


        valid_f0 = f0[f0 > 0]
        print("最小f0", valid_f0.min())

        f0_floor = np.quantile(valid_f0, q=f0_quantile)
        f0_floor = max(f0_floor, self.f0_min)

        max_harm = int(f_nyq // f0_floor) + 1

        print("f0_floor", f0_floor)
        print("max_harm", max_harm)

        freqs_exp = freqs[:, None]      # [n_freq, 1]
        f0_exp = f0[None, :]            # [1, n_frames_stft]

        k_vals = np.arange(max_harm + 1, dtype=np.float64)  # [K]
        low_bound = np.clip((2 * k_vals - 1) / 2, a_min=0.0, a_max=None)  # [K]
        high_bound = (2 * k_vals + 1) / 2                                 # [K]

        low_f = low_bound[:, None, None] * f0_exp[None, :]   # [K, 1, n_frames]
        high_f = high_bound[:, None, None] * f0_exp[None, :] # [K, 1, n_frames]

        freqs_broad = freqs_exp[None, :, :]  # [1, n_freq, 1]
        in_band = (freqs_broad >= low_f) & (freqs_broad < high_f)  # [K, n_freq, n_frames]

        invalid_mask = (f0_exp <= 0)  # [1, n_frames]
        in_band[0] = np.where(invalid_mask, True, in_band[0])
        in_band[1:] = np.where(invalid_mask, False, in_band[1:])

        k_grid = k_vals[:, None, None]  # [K, 1, 1]
        assigned_k = np.sum(k_grid * in_band, axis=0).astype(int)  # [n_freq, n_frames]

        sub_audios = np.zeros((max_harm + 1, n_samples), dtype=np.float64)
        for k in range(max_harm + 1):
            mask = (assigned_k == k)  # [n_freq, n_frames]
            if not np.any(mask):
                continue
            Z_band = Zxx * mask
            _, x_rec = istft(
                Z_band,
                fs=self.sr,
                window='hann',
                nperseg=win_length,
                noverlap=win_length - self.hop_size,
                nfft=n_fft,
                boundary=None,
                time_axis=-1,
                freq_axis=0
            )
            # print("x_rec",x_rec.shape)
            if len(x_rec) > n_samples:
                x_rec = x_rec[:n_samples]
            elif len(x_rec) < n_samples:
                x_rec = np.pad(x_rec, (0, n_samples - len(x_rec)), mode='constant')
            sub_audios[k] = x_rec

        return sub_audios
    
    def SSB_base_time_varying_shift_phase(self, x, shift_hz):
        """
        对实值音频信号x应用时变频移
        通过解析信号与瞬时相位调制实现单边带调制SSB

        Args:
            x (np.ndarray): 输入实值音频信号 (n_samples,)。
            shift_hz (np.ndarray): 时变频移序列Hz, 定义在 STFT 帧时间点上(n_frames,)
                                    正值表示向上频移，负值表示向下频移。

        Returns:
            np.ndarray: 频移后的实值音频信号，(n_samples,)
        """
        n_samples = len(x)
        n_frames = len(shift_hz)
        
        t = np.arange(n_samples) / self.sr
        shift_times = np.arange(n_frames) * self.hop_size / self.sr
        instantaneous_shift = np.interp(t, shift_times, shift_hz)
        
        # 计算瞬时相位变化
        phase_shift = 2 * np.pi * np.cumsum(instantaneous_shift) / self.sr
        
        analytic_signal = hilbert(x)
        shifted_analytic = analytic_signal * np.exp(1j * phase_shift)
        
        shifted_analytic = np.real(shifted_analytic)
        return shifted_analytic
    
    def source_target_harmonic_band_mapping(self, tgt_band_idx, tgt_f0, 
                                            src_f0, src_harmonics
    ):
        """
        合成第 tgt_band_idx 个目标谐波轨道
        """
        hop_size = self.hop_size
        audio_length = src_harmonics.shape[1]
        new_band_wave = np.zeros(audio_length)

        tgt_harmonic_f0 = tgt_f0 * (tgt_band_idx + 1)
        
        # 计算映射关系,目标频率对应第几个源谐波
        src_idx_float = np.divide(tgt_harmonic_f0, src_f0)
        #print("src_idx_float",src_idx_float)
        
        src_idx_floor = np.floor(src_idx_float).astype(int)
        weight_down = src_idx_float - src_idx_floor
        weight_up = 1 - weight_down

        # 遍历所有可能的源谐波索引 (src_k)
        # 我们只关心 src_idx_floor 和 src_idx_floor + 1 覆盖的范围
        min_src = max(0, int(np.min(src_idx_floor)))
        max_src = int(np.max(src_idx_floor)) + 1
        # 限制在实际拥有的源谐波数量内
        # print("src_harmonics.shape", src_harmonics.shape)
        max_src = min(max_src, src_harmonics.shape[0] - 1)
        #print("min_src",min_src)
        #print("max_src",max_src)

        for src_k in range(min_src, max_src+1):
            if src_k == 0:
                continue

            # 找出 src_k 需要向上偏移的时间段
            # 此时 src_k == floor(target)，权重由 weight_up 决定
            is_up_neighbor = (src_idx_floor == src_k)
            
            # 找出 src_k 需要向下偏移的时间段
            # 此时 src_k == floor(target) + 1，权重由 weight_down 决定
            is_down_neighbor = (src_idx_floor == src_k - 1)
            
            # 合并 Mask
            active_mask = is_up_neighbor | is_down_neighbor
            if not np.any(active_mask):
                continue

            # 构建具体的音量 Mask 曲线
            current_loudness_mask = np.zeros_like(src_f0)
            current_loudness_mask[is_up_neighbor] = weight_up[is_up_neighbor]
            # 第1个谐波需要特殊处理，不需要由第0频带合成
            if src_k == 1:
                current_loudness_mask[is_down_neighbor] = np.ones_like(weight_down[is_down_neighbor])
            else:
                current_loudness_mask[is_down_neighbor] = weight_down[is_down_neighbor]
            
            segments = find_segments(active_mask)
            
            for start, end in segments:
                if start == end: continue
                
                # 频率残差 = 目标频率 - 源频率
                # 源频率 = src_f0 * (src_k + 1)
                src_f0_part = src_f0[start:end] * (src_k)
                tgt_f0_part = tgt_harmonic_f0[start:end]
                hz_shift = tgt_f0_part - src_f0_part
                
                src_band_part = src_harmonics[src_k, start * hop_size : end * hop_size]
                mask_slice = current_loudness_mask[start:end]
                
                # 移频
                shifted_wave = self.SSB_base_time_varying_shift_phase(src_band_part, hz_shift)
                target_len = len(shifted_wave)
                mask_samples = get_interpolated_mask(mask_slice, target_len)
                tgt_band_part = shifted_wave * mask_samples
                
                len_proc = len(tgt_band_part)
                len_target = end * hop_size - start * hop_size
                

                L = min(len_proc, len_target)
                target_start = start * hop_size
                new_band_wave[target_start : target_start + L] += tgt_band_part[:L]

        return new_band_wave
    

    def HBMap_shift_f0(self, tgt_f0, src_f0, src_harmonics):
        """
        修改源音频的基频
        """

        print("Synthesizing new harmonics...")
        tgt_f0_max = np.max(tgt_f0)
        tgt_f0_min = np.min(tgt_f0[tgt_f0 > 10])
        if np.isnan(tgt_f0_min): tgt_f0_min = 50
        
        max_harmonic_idx = int((self.sr / 2) // tgt_f0_min) 
        tgt_harmonics = np.zeros((max_harmonic_idx+1, src_harmonics.shape[1]))

        for i in range(0, max_harmonic_idx):
            # print(f"Processing target harmonic {i+1}/{max_harmonic_idx}...")
            
            new_band_wave = self.source_target_harmonic_band_mapping(
                tgt_band_idx=i,
                tgt_f0=tgt_f0,
                src_f0=src_f0,
                src_harmonics=src_harmonics
            )
            
            # output_path = f"new_harmonic_bands2/new_band_{i+1}.wav"
            # sf.write(output_path, new_band_wave, sr, subtype='FLOAT')
            
            tgt_harmonics[i+1] = new_band_wave
        tgt_harmonics[0] = src_harmonics[0]
        print("Mixing final audio...")
        final_audio = np.sum(tgt_harmonics, axis=0)

        return final_audio
    