# -*- coding: utf-8 -*-

import os
import shutil

class FileUtils:
    @staticmethod
    def copy_file(src_file, dest_file):
        # os.system(f"cp {src_file} {dest_file}")
        shutil.copyfile(src_file, dest_file)

    @staticmethod
    def get_files_in(directory):
        file_list = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_list.append(os.path.join(root, file))
        return file_list

    @staticmethod
    # 删除指定文件
    def deleteFileIfExists(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    # 删除指定目录
    def deleteFolderIfExists(folder_path):
        if os.path.exists(folder_path):
            shutil.rmtree(path=folder_path, ignore_errors=True)

    @staticmethod
    def get_filename_without_suffix(file_path):
        file_name = os.path.basename(file_path)
        file_name = os.path.splitext(file_name)[0]
        return file_name

folder_path = "F:/abc"
FileUtils.deleteFolderIfExists(folder_path)