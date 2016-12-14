import os
import copy
import json
import glob

import arff
import xmltodict
from sklearn.cross_validation import train_test_split

def get_ds_list():
    counter = 0
    num_datasets = 10
    dataset_dir = os.path.expanduser("~/.openml/cache/datasets")
    ds_list = []
    for ds_path in glob.glob(dataset_dir + "/*"):
        arff_file = glob.glob(ds_path + "/*.arff")
        if len(arff_file) == 0:
            continue
        arff_file = arff_file[0]
        did = os.path.basename(ds_path)
        # print did, arff_file
        # target_filename = arff_file
        target_filename = ds_path
        if counter < num_datasets:
            ds_list.append(target_filename)
        else:
            break
        counter += 1
    return ds_list


def write_ds_list():
    counter = 0
    num_datasets = 10
    ds_list_fh = open("dataset_list.txt", "w")
    dataset_dir = os.path.abspath("./datasets")
    for ds_path in glob.glob(dataset_dir + "/*"):
        if counter < num_datasets:
            ds_list_fh.write(ds_path+"\n")
        else:
            break
        counter += 1


def process_arff_file(arff_filedir, output_dir, seed):
    did = os.path.basename(arff_filedir)
    arff_data = arff.load(open(arff_filedir + "/dataset.arff", "rb"))
    with open(arff_filedir + "/description.xml") as fh:
        dataset_xml = fh.read()
        description = xmltodict.parse(dataset_xml)["oml:data_set_description"]
    
    target_attribute = description.get("oml:default_target_attribute", None)
    if target_attribute is None:
        print "no default target attribute for %s" % did
        return
    
    try:
        target_index = map(lambda x: x[0], arff_data['attributes']).index(target_attribute)
        # categorical = [False if type(type_) != list else True
        #                for name, type_ in arff_data['attributes']]
        target_categorical = type(arff_data['attributes'][target_index][1]) == list
        assert target_categorical
        target_binary = len(arff_data['attributes'][target_index][1]) == 2
        task = 'binary.classification' if target_binary else 'multiclass.classification'
    except:
        import ipdb;ipdb.set_trace()
        
    # print arff_data.keys()
    # print "target", target_attribute
    # print "task", task
    
    train_data, test_data = train_test_split(arff_data['data'], random_state=seed, test_size=0.1)

    train_arff_data = copy.copy(arff_data)
    train_arff_data['data'] = train_data

    test_arff_data = copy.copy(arff_data)
    test_arff_data['data'] = test_data

    assert len(test_arff_data['data']) + len(train_arff_data['data']) == len(arff_data['data'])

    split_output_dir = os.path.join(os.path.abspath(output_dir), did)
    try:
        os.mkdir(split_output_dir)
    except:
        pass

    print "writing", did
    with open(os.path.join(split_output_dir, "info.json"), 'w') as fh:
         json.dump({
             'target': target_attribute,
             'task': task
         }, fh)
    with open(os.path.join(split_output_dir, "train.arff"), 'wb') as fh:
         arff.dump(train_arff_data, fh)
    with open(os.path.join(split_output_dir, "test.arff"), 'wb') as fh:
         arff.dump(test_arff_data, fh)

seed = 1

for ds_fn in get_ds_list():
    # print ds_fn
    process_arff_file(ds_fn, "./datasets", seed)

write_ds_list()
