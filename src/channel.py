import torch


def add_rayleigh_fading(signal):

    fading = torch.randn_like(signal)

    faded_signal = signal * torch.abs(fading)

    return faded_signal


def add_awgn_noise(signal, snr_db):

    signal_power = torch.mean(signal ** 2)

    snr_linear = 10 ** (snr_db / 10)

    noise_power = signal_power / snr_linear

    noise = torch.randn_like(signal) * torch.sqrt(noise_power)

    noisy_signal = signal + noise

    return noisy_signal


def transmit_signal(signal, snr_db):

    faded_signal = add_rayleigh_fading(signal)

    noisy_signal = add_awgn_noise(
        faded_signal,
        snr_db
    )

    return noisy_signal