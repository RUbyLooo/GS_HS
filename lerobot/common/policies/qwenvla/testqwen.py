from lerobot.common.policies.qwenvla.modeling_qwenvla import QwenVLM
from PIL import Image

def _resize_images(images, size=(224, 224)):
    return [img.resize(size) for img in images]


def main():
    image_list =[]
    vlm = QwenVLM()
    image_path1 = "/home/jiziheng/Desktop/1.png"
    image_path2 = "/home/jiziheng/Desktop/2.png"
    image1 = Image.open(image_path1).convert("RGB")
    image_list.append(image1)
    image2 = Image.open(image_path2).convert("RGB")
    image_list.append(image2)
    captions = ["图像1为第三视角拍摄", "图像2为左腕部相机视角"]
    goal = "pick the banana"

    print("== 生成结构化文本描述 ==")
    desc = vlm.generate_task_description(_resize_images(image_list),captions, goal)
    print(desc)

    # 提取语言 embedding
    print("\n== 提取 prefix embedding ==")
    prefix_emb = vlm.get_prefix_embedding(_resize_images(image_list), captions, goal)
    print(f"Prefix embedding shape: {prefix_emb.shape}")  # [1, L, D]



if __name__ == "__main__":
    main()