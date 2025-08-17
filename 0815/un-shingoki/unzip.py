import os
import zipfile

def unzip_all(src_folder, dst_folder):
    # 创建目标文件夹（如果不存在）
    os.makedirs(dst_folder, exist_ok=True)

    # 遍历文件夹下所有文件
    for root, _, files in os.walk(src_folder):
        for file in files:
            if file.lower().endswith(".zip"):
                zip_path = os.path.join(root, file)
                print(f"正在解压: {zip_path}")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(dst_folder)
                except zipfile.BadZipFile:
                    print(f"⚠️ 无法解压: {zip_path} (坏的zip文件)")

    print("全部解压完成！")

if __name__ == "__main__":
    src = r"C:\Users\79152\Downloads\gpsdata"  # 这里改成存放 zip 的目录
    dst = r"C:\Users\79152\Desktop\github\DX\0815\un-shingoki\gpsdata" # 这里改成解压后的输出目录
    unzip_all(src, dst)
