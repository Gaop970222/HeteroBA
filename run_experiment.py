import subprocess
import time
import datetime


def run_experiments():
    # Dataset-specific trigger types configuration
    dataset_triggers = {
        'IMDB': ['actor', 'director'],
        'DBLP': ['paper'],
        'ACM':['author', 'field'],
    }

    # Experiment parameters
    models = ['HAN',"SimpleHGN",'HGT']
    backdoor_types = ['HeteroBA']
    target_classes = [0, 1, 2]
    base_random_seed = 999

    total_experiments = sum(len(triggers) * len(models) * len(backdoor_types) * len(target_classes) * 3
                            for triggers in dataset_triggers.values())

    start_time = time.time()
    completed_experiments = 0

    print(f"Starting execution of {total_experiments} experiments at {datetime.datetime.now()}")
    print("=" * 80)

    for dataset in dataset_triggers.keys():
        for model in models:
            for backdoor_type in backdoor_types:
                for target_class in target_classes:
                    for trigger_type in dataset_triggers[dataset]:
                        for seed_increment in range(3):
                            random_seed = base_random_seed + seed_increment

                            command = [
                                'python', 'main.py',
                                '--dataset', dataset,
                                '--model_name', model,
                                '--backdoor_type', backdoor_type,
                                '--target_class', str(target_class),
                                '--trigger_type', trigger_type,
                                '--random_seed', str(random_seed)
                            ]

                            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(f"\nStarting experiment at {current_time}")
                            print(
                                f"Configuration: {dataset}, {model}, {backdoor_type}, Class {target_class}, {trigger_type}, Seed {random_seed}")

                            try:
                                subprocess.run(command, check=True)
                                completed_experiments += 1

                                elapsed_time = time.time() - start_time
                                avg_time_per_exp = elapsed_time / completed_experiments
                                remaining_experiments = total_experiments - completed_experiments
                                estimated_remaining_time = avg_time_per_exp * remaining_experiments

                                print(f"Progress: {completed_experiments}/{total_experiments}")
                                print(
                                    f"Estimated remaining time: {datetime.timedelta(seconds=int(estimated_remaining_time))}")
                                print("-" * 80)

                            except subprocess.CalledProcessError as e:
                                print(f"Error executing experiment: {e}")
                                print("Continuing with next experiment...")
                                print("-" * 80)

    end_time = time.time()
    total_time = end_time - start_time

    print("\nExperiment Summary")
    print("=" * 80)
    print(f"Total experiments completed: {completed_experiments}/{total_experiments}")
    print(f"Total execution time: {datetime.timedelta(seconds=int(total_time))}")
    print(f"Average time per experiment: {datetime.timedelta(seconds=int(total_time / completed_experiments))}")
    print("=" * 80)


if __name__ == "__main__":
    run_experiments()
