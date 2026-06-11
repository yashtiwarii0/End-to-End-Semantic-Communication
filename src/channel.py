import torch


def add_awgn_noise(signal, snr_db):

    signal_power = torch.mean(signal**2)

    snr_linear = 10 ** (snr_db / 10)

    noise_power = signal_power / snr_linear

    noise = torch.randn_like(signal) * torch.sqrt(noise_power)

    noisy_signal = signal + noise

    return noisy_signal