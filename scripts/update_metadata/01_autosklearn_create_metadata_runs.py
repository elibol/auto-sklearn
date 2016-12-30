import argparse
import os
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, required=True,
                        help="File with all datasets.")
    parser.add_argument("--runs-per-dataset", default=1, type=int,
                        help="Number of configuration runs per dataset.")
    parser.add_argument("--output-directory", type=str, required=True,
                        help="Configuration output directory.")
    parser.add_argument("--time-limit", type=int, required=True,
                        help="Total configuration time limit.")
    parser.add_argument("--per-run-time-limit", type=int, required=True,
                        help="Time limit for an individual run.")
    parser.add_argument("--ml-memory-limit", type=int, required=True,
                        help="Memory limit for the target algorith run.")
    parser.add_argument('--resampling-strategy',
                        type=str,
                        default="partial-cv",
                        help='Resampling strategy.',
                        choices=['holdout', 'holdout-iterative-fit', 'cv', 'nested-cv', 'partial-cv'])
    parser.add_argument("--data-format", type=str, default="automl-competition-format",
                        help="format of data.",
                        choices=["automl-competition-format", "arff"])
    parser.add_argument("--metric", type=str, required=False, default="acc",
                        help="Metric to use.")
    parser.add_argument('-e', '--include-estimators',
                        nargs='*',
                        default=None,
                        type=str,
                        help='Only use these estimators inside auto-sklearn')
    parser.add_argument('-p', '--include-preprocessors',
                        nargs='*',
                        default=None,
                        type=str,
                        help='Only use these preprocessors inside auto-sklearn')
    args = parser.parse_args()

    datasets_dir = os.path.abspath(os.path.dirname(args.datasets))
    datasets = []
    with open(args.datasets, 'r') as fh:
        for line in fh:
            line = line.strip()
            dataset = os.path.join(datasets_dir, line)
            datasets.append(dataset)

    commands = []
    for num_run in range(args.runs_per_dataset):
        for dataset in datasets:
            if args.data_format == 'arff':
                # load target and task data from info.json
                # e.g. {'target': 'y', 'task': 'binary.classification'}
                # TODO: these can be inferred from openml datasets
                with open(dataset + "/info.json") as fh:
                    info = json.load(fh)
                target = info['target']
                task = info['task']
            
            if dataset[-1] == "/":
                dataset = dataset[:-1]
            dataset_name = os.path.basename(dataset)
            output_directory = os.path.join(args.output_directory, dataset_name)
            try:
                os.makedirs(output_directory)
            except:
                pass
            # '-e lda xgradient_boosting qda extra_trees decision_tree gradient_boosting k_nearest_neighbors multinomial_nb libsvm_svc gaussian_nb random_forest bernoulli_nb ' \
            # '-p polynomial pca ' \
            command = 'autosklearn --output-dir %s ' \
                      '--temporary-output-directory %s ' \
                      '--seed %d ' \
                      '--time-limit %d ' \
                      '--per-run-time-limit %d ' \
                      '--ml-memory-limit %d ' \
                      '--ensemble-size 1 ' \
                      '--metalearning-configurations 0 ' \
                      '--target %s ' \
                      '--task %s ' \
                      '--dataset %s' \
                      '--resampling-strategy %s ' \
                      '--data-format %s ' \
                      '--metric %s ' \
                      '-e %s ' \
                      '-p %s ' \
                      % (output_directory,
                         output_directory,
                         num_run * 1000,
                         args.time_limit,
                         args.per_run_time_limit,
                         args.ml_memory_limit,
                         target,
                         task,
                         dataset,
                         args.resampling_strategy,
                         args.data_format,
                         args.metric,
                         args.include_estimators,
                         args.include_preprocessors,)
            commands.append(command)

    commands_file = os.path.join(args.output_directory, 'commands.txt')
    with open(commands_file, 'w') as fh:
        for command in commands:
            fh.write("%s\n" % command)
