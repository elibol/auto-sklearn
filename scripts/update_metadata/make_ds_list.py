import os
import glob

counter = 0
num_datasets = 10
ds_list_fh = open("dataset_list.txt", "w")
dataset_dir = os.path.expanduser("~/.openml/cache/datasets")
for ds_path in glob.glob(dataset_dir + "/*"):
    arff_file = glob.glob(ds_path + "/*.arff")
    if len(arff_file) == 0:
        continue
    arff_file = arff_file[0]
    did = os.path.basename(ds_path)
    print did, arff_file

    # target_filename = arff_file
    target_filename = ds_path
    if counter < num_datasets:
        ds_list_fh.write(target_filename+"\n")
    else:
        break
    counter += 1
    
