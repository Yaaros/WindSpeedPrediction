# coding=utf-8
import os


def get_project_root():
    """获取当前文件所在的项目根目录"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def get_target_directory(subdir):
    """获取项目根目录下指定子目录的完整路径"""
    root_dir = get_project_root()
    target_dir = os.path.join(root_dir, 'dataset', subdir)
    str.replace(target_dir, '\\', '\\\\')
    if not os.path.exists(target_dir):
        raise NotADirectoryError(f"The directory {target_dir} does not exist.")

    return target_dir