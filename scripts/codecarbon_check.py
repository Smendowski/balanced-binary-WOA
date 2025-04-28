from codecarbon import EmissionsTracker

output_file = "codecarbon_example.csv"

tasks = ["task_1", "task_2", "task_3", "task_4", "task_5"]

for task in tasks:
    tracker = EmissionsTracker(project_name=task, output_file=output_file)
    tracker.start()

    for _ in range(10**6):
        _ = sum([i for i in range(100)])

    tracker.stop()
