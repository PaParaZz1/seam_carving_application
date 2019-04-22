import cv2
import numpy as np

def gradient_L1(x):
    x = np.copy(x).astype(np.float32)
    x = cv2.GaussianBlur(x, (7, 7), 0)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    return (np.abs(sobel_x) + np.abs(sobel_y))*255


def gradient_canny(x):
    img = (np.copy(x)*255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (7, 7), 0)
    canny = cv2.Canny(img, 50, 150)
    return canny


def main(input_path, scale_threshold=0.4):
    img = cv2.imread(input_path)
    img = (img/255.).astype(np.float32)
    H, W, C = img.shape
    noise_img = img + np.random.randn(H, W, 3)*0.1
    noise_img = noise_img.clip(0, 1)
    scale_img = img*scale_threshold / img.max()
    scale_noise_img = scale_img + np.random.randn(H, W, 3)*0.1
    scale_noise_img = scale_noise_img.clip(0, 1)

    name = input_path.split('.')[0]
    cv2.imwrite(name+"_noise.png", noise_img*255)
    cv2.imwrite(name+"_scale.png", scale_img*255)
    cv2.imwrite(name+"_scale_noise.png", scale_noise_img*255)

    cv2.imwrite(name+"_L1.png", gradient_L1(img))
    cv2.imwrite(name+"_canny.png", gradient_canny(img))
    cv2.imwrite(name+"_noise_L1.png", gradient_L1(noise_img))
    cv2.imwrite(name+"_noise_canny.png", gradient_canny(noise_img))
    cv2.imwrite(name+"_scale_L1.png", gradient_L1(scale_img))
    cv2.imwrite(name+"_scale_canny.png", gradient_canny(scale_img))
    cv2.imwrite(name+"_scale_noise_L1.png", gradient_L1(scale_noise_img))
    cv2.imwrite(name+"_scale_noise_canny.png", gradient_canny(scale_noise_img))


if __name__ == "__main__":
    main("jun1.jpg")
