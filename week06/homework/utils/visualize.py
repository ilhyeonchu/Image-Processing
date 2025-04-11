import matplotlib.pyplot as plt

def visualize_gradient_outputs(image, grad_x, grad_y, magnitude):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.title("Input image")
    plt.axis("off")
    plt.imshow(image, cmap="gray")

    plt.subplot(1, 4, 2)
    plt.title("x-direction gradient")
    plt.axis("off")
    plt.imshow(grad_x, cmap="gray")

    plt.subplot(1, 4, 3)
    plt.title("y-direction gradient")
    plt.axis("off")
    plt.imshow(grad_y, cmap="gray")

    plt.subplot(1, 4, 4)
    plt.title("Gradient magnitude")
    plt.axis("off")
    plt.imshow(magnitude, cmap="gray")
    plt.show()

def visualize_dog_filters(dog_x, dog_y):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("x-direction DoG filter")
    plt.axis("off")
    plt.imshow(dog_x, cmap="gray", interpolation="bicubic")

    plt.subplot(1, 2, 2)
    plt.title("x-direction DoG filter")
    plt.axis("off")
    plt.imshow(dog_y, cmap="gray", interpolation="bicubic")

    plt.show()

def visualize_log_filters(log_x, log_y):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("x-direction LoG filter")
    plt.axis("off")
    plt.imshow(log_x, cmap="gray", interpolation="bicubic")

    plt.subplot(1, 3, 2)
    plt.title("x-direction LoG filter")
    plt.axis("off")
    plt.imshow(log_y, cmap="gray", interpolation="bicubic")

    plt.subplot(1, 3, 3)
    plt.title("LoG filter")
    plt.axis("off")
    plt.imshow(log_x+log_y, cmap="gray", interpolation="bicubic")

    plt.show()

def visualize_log_output(image, log_output):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Input image")
    plt.axis("off")
    plt.imshow(image, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("LoG filtering output")
    plt.axis("off")
    plt.imshow(log_output, cmap="gray")

    plt.show()

def visualize_zero_crossing_output(image, log_output, zero_crossing):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Input image")
    plt.axis("off")
    plt.imshow(image, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title("LoG filtering output")
    plt.axis("off")
    plt.imshow(log_output, cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title("Find zero crossing")
    plt.axis("off")
    plt.imshow(zero_crossing, cmap="gray")
    plt.show()

def visualize_binarization_output(image, gradient_magnitude, binarization_output):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Input image")
    plt.axis("off")
    plt.imshow(image, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title("DoG magnitude")
    plt.axis("off")
    plt.imshow(gradient_magnitude, cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title("Binarization")
    plt.axis("off")
    plt.imshow(binarization_output, cmap="gray")
    plt.show()