from PIL import Image

# 对图片进行等比缩放的代码
def keep_image_size_open(path, size=(256, 256)):
    img = Image.open(path)
    max_size = max(img.size)  # 取最长边
    mask = Image.new('RGB', (max_size, max_size), (0, 0, 0,))  # 针对最长边进行mask掩码,（0，0，0）表示黑色掩码
    mask.paste(img, (0, 0))  # 原图粘到mask上，左上角位置
    mask = mask.resize(size)
    return mask
