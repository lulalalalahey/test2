import os
import random
import shutil

# 输入图片的文件夹路径（即 sad 文件夹）
source_dir = 'F:\\github\\fer2013\\original_data_after_augmentation\\original\\sad1'

# 输出选取的图片保存路径
destination_dir = 'F:\\github\\fer2013\\original_data_after_augmentation\\original\\sad'

# 确保输出路径存在
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# 收集 sad 文件夹中所有图片的路径
all_images = [os.path.join(source_dir, img_name) for img_name in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, img_name))]

# 检查是否有足够的图片
if len(all_images) < 3500:
    raise ValueError(f"Not enough images. Found only {len(all_images)} images, but need 3500.")

# 随机选取3500张图片
selected_images = random.sample(all_images, 3500)

# 将选中的图片复制到目标文件夹
for img_path in selected_images:
    # 获取图片的文件名
    img_name = os.path.basename(img_path)
    # 复制图片到目标文件夹
    shutil.copy(img_path, os.path.join(destination_dir, img_name))

print(f"Successfully selected and copied 3500 images to {destination_dir}.")
