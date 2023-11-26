from environment import *

def fixed_rotation(image, angles=[90, 180, 270]):
    """Rotate the image by fixed angles."""
    return [image.rotate(angle) for angle in angles]

def fixed_flip(image):
    """Flip the image horizontally and vertically."""
    horizontal_flip = ImageOps.mirror(image)
    vertical_flip = ImageOps.flip(image)
    return [horizontal_flip]

def fixed_brightness(image, factors=[0.99, 1.01]):
    """Adjust the brightness of the image by fixed factors."""
    image = image.convert('RGB')
    enhancer = ImageEnhance.Brightness(image)
    return [enhancer.enhance(factor) for factor in factors]

def fixed_contrast(image, factors=[0.99, 1.01]):
    """Adjust the contrast of the image by fixed factors."""
    image = image.convert('RGB')
    enhancer = ImageEnhance.Contrast(image)
    return [enhancer.enhance(factor) for factor in factors]

def fixed_saturation(image, factors=[0.99, 1.01]):
    """Adjust the saturation of the image by fixed factors."""
    image = image.convert('RGB')
    enhancer = ImageEnhance.Color(image)
    return [enhancer.enhance(factor) for factor in factors]

def fixed_hue(image, shifts=[0.01, -0.01]):
    """Shift the hue of the image by fixed amounts."""
    images = []
    for shift in shifts:
        img = image.convert('HSV')
        h, s, v = img.split()
        np_h = np.array(h, dtype=np.uint8)
        np_h = (np_h + int(shift * 255)) % 255
        h = Image.fromarray(np_h, 'L')
        img = Image.merge('HSV', (h, s, v)).convert('RGB')
        images.append(img)
    return images

# Example usage:
# image_path = "path_to_your_image.jpg"
# image = Image.open(image_path)

# rotated_images = fixed_rotation(image)
# flipped_images = fixed_flip(image)
# brightness_images = fixed_brightness(image)
# contrast_images = fixed_contrast(image)
# saturation_images = fixed_saturation(image)
# hue_images = fixed_hue(image)

# To display or save the augmented images, you can loop through each list and use the .show() or .save() methods.


