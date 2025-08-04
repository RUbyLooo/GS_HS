import os
import json

# 根路径
IMG_ROOT = "/sda1/lerobot/data/single_arm_apple_0529_1/images/observation.images.3rd"
OUTPUT_PATH = "/sda1/lerobot/data/single_arm_apple_0529_1/meta/episodes.jsonl"

def main():
    episode_dirs = sorted([
        d for d in os.listdir(IMG_ROOT)
        if d.startswith("episode_") and os.path.isdir(os.path.join(IMG_ROOT, d))
    ])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for idx, dirname in enumerate(episode_dirs):
            episode_path = os.path.join(IMG_ROOT, dirname)
            # 统计图片帧数
            frame_count = len([
                name for name in os.listdir(episode_path)
                if name.endswith(".png")
            ])

            episode_entry = {
                "episode_index": idx,
                "tasks": ["to grasp the apple with arm"],
                "length": frame_count
            }
            f.write(json.dumps(episode_entry) + "\n")

    print(f"✅ Successfully rebuilt episodes.jsonl with {len(episode_dirs)} episodes.")
    print(f"📄 Output written to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
