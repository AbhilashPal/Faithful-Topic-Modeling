import argparse

from src.utils import load_20newsgroups_and_save_csv,create_folders_if_not_exist

def main():
    parser = argparse.ArgumentParser(description="Save 20newsAG.")
    parser.add_argument("--path", type=str, help="Name of the folder to create.")
    args = parser.parse_args()
    create_folders_if_not_exist(args.path)
    load_20newsgroups_and_save_csv(args.path)

if __name__ == "__main__":
    main()