import os
import argparse

def clean(target_dir):
    rootPath = os.getcwd()
    target_path = os.path.abspath(target_dir)
    os.chdir(target_path)
    clean_file = os.listdir(os.getcwd())
    for i in clean_file:
        clean_path = os.path.join(target_path,i)
        os.chdir(clean_path)
        os.system('sudo find ./ -name "._*" -depth -exec rm {} \;')
        #clean_file2 = os.listdir(os.getcwd())
        #for n in clean_file2:
        #   clean_path2 = os.path.join(clean_path,n)
        #   os.chdir(clean_path2)
        #   print(os.getcwd())
#           os.system('sudo find ./ -name "._*" -depth -exec rm {} \;')






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean the hiding file')
    parser.add_argument('target_folder', help='Path to folder where to be clean.')
    args = parser.parse_args()
    clean(args.target_folder)
