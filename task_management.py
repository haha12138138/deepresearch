from typing import List, Optional, Literal
from data_models import Task_Description


class Task:
    _id_tracker = 0

    def __init__(self, task_id, description):
        self.task_id = task_id
        self.description: Task_Description = description
        self.status: Literal['QUEUED', 'RUNNING', 'SUCCEEDED', 'ERROR'] = "QUEUED"
        self.result: Optional[str] = None

    @classmethod
    def construct_from_description(cls, description: Task_Description):
        new_task = cls(cls._id_tracker, description)
        cls._id_tracker += 1
        return new_task

    def __str__(self):
        return self.description.action


class Task_Queue:
    def __init__(self):
        self.error_queue: List[Task] = []
        self.finish_queue: List[Task] = []
        self.unfinished_queue: List[Task] = []
        self.current_task = None

    def add_new_tasks(self, task_list: List[Task]):
        # task_list = list(map(lambda x: Task.construct_from_description(x), task_descriptions.l))
        self.unfinished_queue = task_list

    def issue_task(self):
        self.current_task = self.unfinished_queue.pop(0)
        return self.current_task

    def commit_task(self, task: Task):
        if task.status == 'ERROR':
            self.error_queue.append(task)
        elif task.status == "SUCCEEDED":
            self.finish_queue.append(task)
        else:
            raise ValueError(f"the status of committed task is neither SUCCEEDED nor ERROR. received: {task.status}")

    def __str__(self):
        ans = ""
        l = map(lambda x: x.__str__(), self.error_queue)
        ans += " status: Error\r".join(l)
        l = map(lambda x: x.__str__(), self.finish_queue)
        ans += " status: OK\r".join(l)
        l = map(lambda x: x.__str__(), self.unfinished_queue)
        ans += " status: pending\r".join(l)
        ans += " status: pending\r"
        ans += f"{self.current_task.__str__()} status: executing"
        return ans

    def __iter__(self):
        # The iterator is the queue itself
        return self

    def __next__(self) -> Task:
        # If no more tasks, signal the end of iteration
        if not self.unfinished_queue:
            raise StopIteration
        return self.issue_task()
