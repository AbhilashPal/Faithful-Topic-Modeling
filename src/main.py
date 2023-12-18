import argparse

from src.utils import create_folders_if_not_exist, read_csv_column
from src.calculate_comprehensiveness import dump_comprehensiveness_results

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Create a folder if it doesn't exist.")
    parser.add_argument("--path", type=str, help="Name of the path to create.")
    parser.add_argument("--dataset", type=str, help="CSV dataset with the data in one column.")
    parser.add_argument("--column", type=str, help="CSV dataset column.")
    parser.add_argument("--k", type=int, help="Top k topics to run the ablation for.")
    parser.add_argument("--keybert", action="store_true", help="Use KeybertInspired or Not.")
    # Parse the command-line arguments
    args = parser.parse_args()
    # Call the function to create the folder
    create_folders_if_not_exist(args.path+"Temporary_Results")
    create_folders_if_not_exist(args.path+"Temporary_Results/Base_Results")
    create_folders_if_not_exist(args.path+"Temporary_Results/Topic_Results")
    create_folders_if_not_exist(args.path+"Processed_Results")

    docs = read_csv_column(csv_file=args.dataset,column_name=args.column)

    #call comprehensiveness run
    dump_comprehensiveness_results(docs,args.k,args.path,use_keybert=args.keybert)

if __name__ == "__main__":
    main()