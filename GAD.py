# Guided Anisotropic Diffusion algorithm
# Rodrigo Caye Daudt
# https://rcdaudt.github.io
#
# Caye Daudt, Rodrigo, Bertrand Le Saux, Alexandre Boulch, and Yann Gousseau. "Guided anisotropic diffusion and iterative learning for weakly supervised change detection." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 0-0. 2019.

import torch


def g(x, K=5):
    return 1.0 / (1.0 + (torch.abs((x * x)) / (K * K)))


def get_image_gradients(image):
    dv = image[:, :, 1:, 1:-1] - image[:, :, :-1, 1:-1]
    dh = image[:, :, 1:-1, 1:] - image[:, :, 1:-1, :-1]
    return dv, dh


def diffusion_coefficient(gradient_v, gradient_h, K):
    cv = g(torch.mean(gradient_v, 1), K)
    ch = g(torch.mean(gradient_h, 1), K)
    return cv, ch


def diffuse(image, lambda_, K, coeffs=None, return_coeffs=False):
    dv, dh = get_image_gradients(image)
    if coeffs is None:
        cv, ch = diffusion_coefficient(dv, dh, K)
    else:
        cv, ch = coeffs
    for channel in range(image.size(1)):
        image[:, channel, 1:-1, 1:-1] += lambda_ * (
            cv[:, 1:, :] * dv[:, channel, 1:, :]
            - cv[:, :-1, :] * dv[:, channel, :-1, :]
            + ch[:, :, 1:] * dh[:, channel, :, 1:]
            - ch[:, :, :-1] * dh[:, channel, :, :-1]
        )
    if return_coeffs:
        return image, cv, ch
    else:
        return image


def anisotropic_diffusion(
    first_guide, second_guide, input_image, N=500, lambda_=0.24, K=5, is_log=True, verbose=False
):

    if is_log:
        input_image = torch.exp(input_image)

    with torch.no_grad():
        for t in range(N):
            if verbose:
                print("Iteration {}".format(t))

            # Perform diffusion on the first guide
            first_guide, cv1, ch1 = diffuse(first_guide, lambda_, K, return_coeffs=True)

            # Perform diffusion on the second guide (if specified)
            if second_guide is not None:
                second_guide, cv2, ch2 = diffuse(second_guide, lambda_, K, return_coeffs=True)
                cv = torch.min(cv1, cv2)
                ch = torch.min(ch1, ch2)
            else:
                cv, ch = cv1, ch1

            input_image = diffuse(input_image, lambda_, K, coeffs=(cv, ch))

    return input_image
