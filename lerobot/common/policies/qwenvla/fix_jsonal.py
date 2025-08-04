import re
import json

def replace_episode_descriptions(input_path, output_path):
    pattern = re.compile(r'episode_\d+')
    if input_path == output_path:
        raise ValueError("❌ Input and output paths must be different to prevent data loss.")

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)

            # 替换所有 string 和 list[str] 中的 episode_*
            for key, value in data.items():
                if isinstance(value, str):
                    data[key] = pattern.sub("to grasp the banana with arm", value)
                elif isinstance(value, list):
                    data[key] = [
                        pattern.sub("to grasp the banana with arm", v) if isinstance(v, str) else v
                        for v in value
                    ]

            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    input_file = "/sda1/lerobot/data/single_arm_banana_0529_1/meta/episodes.jsonl"
    output_file = "/sda1/lerobot/data/single_arm_banana_0529_1/meta/episodes_.jsonl"
    replace_episode_descriptions(input_file, output_file)
    print(f"✅ Done! Saved to: {output_file}")
