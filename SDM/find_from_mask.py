import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def extract_object(image_path, mask_path, output_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        raise ValueError("无法读取图像或mask，请检查路径")

    extracted_object = cv2.bitwise_and(image, image, mask=mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("没有找到物体轮廓")

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_object = extracted_object[y:y + h, x:x + w]

    cv2.imwrite(output_path, cropped_object)
    print(f"✅ 图像保存成功：{output_path}")
    return cropped_object


def find_base_mask(data, prompt, o_path, fold_p, topn=150):
    # 将结果固定保存到 results/views/
    org_path = os.path.join(o_path, "results", "views")
    os.makedirs(org_path, exist_ok=True)
    print(f"输出目录: {org_path}")

    df = pd.DataFrame(data, columns=['Image', 'Value', 'Path'])
    top_data = df.nlargest(topn, 'Value').values.tolist()

    for item in top_data:
        folder_name = os.path.basename(os.path.dirname(item[2]))
        mask_file = item[0]
        mask_path = os.path.join(fold_p, folder_name, mask_file)
        image_path = os.path.join(fold_p, folder_name + ".jpg")
        opt_path = os.path.join(org_path, f"{folder_name}_{mask_file}")

        print(f"处理: {mask_path}")
        print(f"原图: {image_path}")
        print(f"输出: {opt_path}")

        extracted_object = extract_object(image_path, mask_path, opt_path)

    return org_path
