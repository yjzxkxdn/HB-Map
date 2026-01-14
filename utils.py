import numpy as np
from scipy.interpolate import interp1d

def find_segments(mask):
    """
    找出布尔数组中所有连续 True 片段的起始和结束索引（包含两端）。
    
    参数:
        mask (np.ndarray): 布尔型或 0/1 的一维数组。
    
    返回:
        List[Tuple[int, int]]: 每个元组表示一个连续 True 段的 [start, end]（闭区间）。
    """
    # 确保输入是布尔数组
    mask = np.asarray(mask, dtype=bool)
    
    # 在前后各加一个 False，便于检测边界
    padded = np.concatenate([[False], mask, [False]])
    
    # 计算差分：+1 表示 True 段开始，-1 表示 True 段结束
    diff = np.diff(padded.astype(int))
    
    # 起始位置：diff == 1 的索引（即从 False 到 True）
    starts = np.where(diff == 1)[0]
    
    # 结束位置：diff == -1 的索引减 1（因为 padded 多了一个前导 False）
    ends = np.where(diff == -1)[0] - 1
    
    # 组合成列表
    return list(zip(starts, ends))


def get_interpolated_mask(mask_frames, target_len):
    """将帧级 Mask 插值到 样本级 Mask，并进行数值截断防止爆音"""
    n_frames = len(mask_frames)
    if n_frames < 2:
        # 如果片段太短无法插值，直接填充常量
        return np.full(target_len, mask_frames[0])
    
    x_old = np.linspace(0, n_frames - 1, n_frames)
    x_new = np.linspace(0, n_frames - 1, target_len)
    
    # 关键修正：限制插值范围，防止extrapolate产生极端值
    interpolator = interp1d(x_old, mask_frames, kind='linear', fill_value="extrapolate")
    mask_samples = interpolator(x_new)
    
    # 强制截断到 [0, 1] 区间，防止任何可能的数值爆炸
    return np.clip(mask_samples, 0.0, 1.0)