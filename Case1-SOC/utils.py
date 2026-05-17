import numpy as np

def compute_soc(time_arr: np.ndarray, current_arr: np.ndarray, c_nom: float = 2.0) -> np.ndarray:
    # 计算各步时间增量
    dt = np.diff(time_arr, prepend=time_arr[0])
    # 安时积分: A * s 转换为 Ah
    ah = np.cumsum(current_arr * dt) / 3600.0
    soc = ah / c_nom
    return soc