import os
import pandas as pd

def txt_to_csv_all(src_folder, dst_folder):
    # 创建目标文件夹（如果不存在）
    os.makedirs(dst_folder, exist_ok=True)

    # 遍历源文件夹
    for root, _, files in os.walk(src_folder):
        for file in files:
            if file.lower().endswith(".txt"):
                txt_path = os.path.join(root, file)
                csv_name = os.path.splitext(file)[0] + ".csv"
                csv_path = os.path.join(dst_folder, csv_name)

                print(f"正在转换: {txt_path} -> {csv_path}")
                try:
                    # 读取 txt 并保存为 csv
                    df = pd.read_csv(txt_path, sep=None, engine="python")  # sep=None 自动推断分隔符
                    df.to_csv(csv_path, index=False)
                except Exception as e:
                    print(f"⚠️ 无法转换 {txt_path}: {e}")

    print("全部转换完成！")

if __name__ == "__main__":
    src = r"C:\Users\79152\Desktop\github\DX\0815\un-shingoki\gpsdata"    # 这里改成存放 txt 的目录
    dst = r"C:\Users\79152\Desktop\github\DX\0815\un-shingoki\gpsdata-csv"    # 这里改成输出 csv 的目录
    txt_to_csv_all(src, dst)
