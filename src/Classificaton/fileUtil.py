import os
import re

def get_file_name(path, pre = "model", ext = ".pth"):
    pattern = re.compile(rf"^{re.escape(pre)}(\d+){re.escape(ext)}$")
    max_num = 0

    for file in os.listdir(path):
        match = pattern.match(file)
        if match:
            num = int(match.group(1))
            max_num = max(max_num, num)
    next_num = max_num + 1
    next_filename = f"{pre}{next_num}{ext}"
    print(f"File Saved As: {next_filename}")
    return os.path.join(path, next_filename)

get_file_name("weights/", "model", ".pth")




