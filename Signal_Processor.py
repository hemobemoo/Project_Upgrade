import numpy as np
import pywt
from scipy.signal import butter, filtfilt

class SignalAPIProcessor:
    def __init__(self, fs=100.0, lowcut=0.5, highcut=45.0, order=4, target_len=1000):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.target_len = target_len

        self.means = np.array([-8.797371736859255e-05, -8.813923356093718e-05, 2.0138948488197318e-07,
                               8.800947941879742e-05, -4.07881299240516e-05, -4.115890291417214e-05,
                               5.807763620681135e-05, -1.1556609195845202e-05, -4.839426167934671e-05,
                               -9.133048638424147e-05, -0.00012264618564742757, -0.0001915697616177152])
        self.stds = np.array([0.14488022574881212, 0.14412726783411506, 0.13705478487169195,
                              0.1272077201355749, 0.12125352159483123, 0.12039017273841657,
                              0.20634558877626064, 0.31987098439992, 0.30852413088102504,
                              0.2759407463310302, 0.24339637364402752, 0.19542406921362407])

    def _bandpass_filter(self, data):
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(self.order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=0)

    def _wavelet_denoising(self, data, wavelet='db4', level=2):
        cleaned_data = np.zeros_like(data)
        for lead in range(data.shape[1]):
            sig = data[:, lead]
            coeffs = pywt.wavedec(sig, wavelet, level=level)
            detail_coeffs = coeffs[-1]
            sigma = np.median(np.abs(detail_coeffs)) / 0.6745
            uthresh = (sigma * np.sqrt(2 * np.log(len(sig)))) * 0.5

            new_coeffs = [coeffs[0]]
            for i in range(1, len(coeffs)):
                c = coeffs[i]
                c_thresholded = np.sign(c) * np.maximum(np.abs(c) - uthresh, 0)
                new_coeffs.append(c_thresholded)

            cleaned_data[:, lead] = pywt.waverec(new_coeffs, wavelet)
        return cleaned_data

    def process_for_api(self, raw_signal):
        """
        raw_signal: numpy array shape (T, C) where C is number of leads.
        Returns: normalized array shape (1, target_len, C)
        """
        if raw_signal.ndim == 1:
            raw_signal = raw_signal[:, np.newaxis]

        n_leads = raw_signal.shape[1]

        # Truncate or pad statistics to match actual lead count
        means = self.means[:n_leads] if n_leads <= len(self.means) else np.pad(self.means, (0, n_leads - len(self.means)), mode='edge')
        stds  = self.stds[:n_leads]  if n_leads <= len(self.stds)  else np.pad(self.stds,  (0, n_leads - len(self.stds)),  mode='edge')

        filtered   = self._bandpass_filter(raw_signal)
        denoised   = self._wavelet_denoising(filtered)
        normalized = (denoised - means) / (stds + 1e-7)

        if len(normalized) > self.target_len:
            normalized = normalized[:self.target_len]
        elif len(normalized) < self.target_len:
            pad = np.zeros((self.target_len - len(normalized), n_leads))
            normalized = np.concatenate([normalized, pad], axis=0)

        return normalized[np.newaxis, :, :]

    def process_stream_window(self, window, fs=None):
        """Process a single streaming window with the same pipeline."""
        if fs is not None:
            self.fs = fs
        return self.process_for_api(window)