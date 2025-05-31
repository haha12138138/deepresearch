from typing import List, Optional, Literal # Added Literal here
from data_models import Goal_Description, Task_Description_List
from task_management import Task_Queue, Task


class Goal_Info:
    def __init__(self, description: Goal_Description):
        self.description = description
        self.status: Literal['QUEUED', 'RUNNING', 'SUCCEEDED', 'ERROR'] = "QUEUED"
        self.findings: List[str] = [] # Corrected type hint from [str] to List[str]
        self.summary: str = ""
        self.task_queue: Task_Queue = Task_Queue()

    def __str__(self):
        return f"Description: {self.description}, findings {self.findings}"

    @classmethod
    def construct_from_description(cls, description: Goal_Description):
        new_goal = cls(description)
        return new_goal


class Snapshot_Node:
    def __init__(self, goal_info: Goal_Info): # Goal_Info is defined above
        self.this: Optional[int] = None
        self.parent: Optional[int] = None
        self.children: List[Optional[int]] = []
        self.goal: Goal_Info = goal_info

    def set_parent_id(self, parent_idx):
        self.parent = parent_idx

    def set_self_id(self, self_idx):
        self.this = self_idx

    def add_chld(self, chld_idx):
        self.children.append(chld_idx)


class Research_Snapshot:
    @classmethod
    def create_goal_node(cls, goal_description: Goal_Description) -> Snapshot_Node: # Snapshot_Node is defined above
        return Snapshot_Node(Goal_Info.construct_from_description(goal_description))

    @classmethod
    def create_task_list(cls, task_descriptions: Task_Description_List) -> List[Task]:
        task_list = list(map(lambda x: Task.construct_from_description(x), task_descriptions.l))
        return task_list

    @classmethod
    def initialize_from_usr_request(cls, description: Goal_Description) -> "Research_Snapshot":
        root_node = cls.create_goal_node(description)
        snapshot = cls()

        snapshot.insert_node(root_node)
        snapshot.active_cursor = 0
        return snapshot

    def __init__(self):
        self.active_cursor: Optional[int] = None
        self.nodes: List[Snapshot_Node] = [] # Snapshot_Node is defined above
        self._idx_tracker = 0

    def insert_node(self, node: Snapshot_Node): # Snapshot_Node is defined above
        p_idx = None if self.active_cursor is None else self.get_current_node().this
        n_idx = self._idx_tracker
        node.set_parent_id(p_idx)
        node.set_self_id(n_idx)
        if p_idx is not None:
            self.nodes[p_idx].children.append(n_idx)
        else:
            node.goal.status = "RUNNING"

        self.nodes.append(node)
        self._idx_tracker +=1

    def get_parent_idx(self) -> Optional[int]:
        if self.active_cursor is not None:
            return self.nodes[self.active_cursor].parent
        else:
            return None

    def get_node_idx(self) -> Optional[int]:
        return self._idx_tracker

    def get_current_node(self) -> Snapshot_Node: # Snapshot_Node is defined above
        if self.active_cursor is not None:
            return self.nodes[self.active_cursor]
        else:
            return None

    def explore(self) -> Optional[Snapshot_Node]: # Snapshot_Node is defined above
        """
        this function will select a QUEUED sub node to explore. if all the subnodes are finished, will return the parent node
        if this is already the root node. it will return None
        :return: node to be processed
        """
        current_node = self.get_current_node()
        for chld in current_node.children:
            if self.nodes[chld].goal.status == "QUEUED":
                chld_node = self.nodes[chld]
                chld_node.goal.status = "RUNNING"
                self.active_cursor = chld
                return chld_node

        current_node.goal.status = "SUCCEEDED"
        parent_node = self.nodes[current_node.parent] if current_node.parent is not None else current_node
        return parent_node

    def insert_tasks(self, task_descriptions: Task_Description_List):
        """
        You can use this function to:
        1. do preparation for more serious task that needs more information like goal splitting
        2. assign jobs like search, RAG ... to get enough information to complete the research goal.
        :param task_descriptions
        :return: None. it will add tasks to the task queue
        """
        node = self.get_current_node()
        # Assuming task_descriptions is already a list of Task objects or can be directly used by add_new_tasks
        # If task_descriptions is Task_Description_List, it should be
        # node.goal.task_queue.add_new_tasks([Task.construct_from_description(d) for d in task_descriptions.l])
        # The original code in structure_predict.py for Research_Snapshot.insert_tasks was:
        # node.goal.task_queue.add_new_tasks(task_descriptions)
        # This implies task_descriptions was already List[Task]
        # However, the method _plan_tasks in Research_Agent returns Task_Description_List
        # and then current_node.goal.task_queue.add_new_tasks([Task.construct_from_description(d) for d in task_desc_list.l]) is called
        # So, insert_tasks should probably expect List[Task] or handle Task_Description_List
        # For now, I will keep it as is, assuming task_descriptions is List[Task] as per its usage in Goal_Info.task_queue
        # If `task_descriptions` is `Task_Description_List` (as it comes from `_plan_tasks`),
        # then it should be processed into `List[Task]` before calling `add_new_tasks` on `Task_Queue`.
        # The current `Task_Queue.add_new_tasks` expects `List[Task]`.
        # The method `Research_Snapshot.create_task_list` does this conversion.
        # It seems `Research_Agent._execute_pending_tasks` calls `task_queue.add_new_tasks` with a list of actual `Task` objects.
        # And `Research_Agent.run_research` calls `current_node.goal.task_queue.add_new_tasks([Task.construct_from_description(d) for d in task_desc_list.l])`
        # Let's assume `insert_tasks` is called with `Task_Description_List` and needs to convert.
        # No, the call in `Research_Snapshot.insert_tasks` is `node.goal.task_queue.add_new_tasks(task_descriptions)`
        # This means task_descriptions parameter for this method should be List[Task].
        # The `Research_Agent` calls `self.snapshot.insert_tasks(tasks)` where tasks are `List[Task]`.
        # Oh, I see, the `Research_Agent.insert_tasks` is a different method signature.
        # The `Research_Snapshot.insert_tasks` expects Task_Description_List based on its usage in `Research_Agent.run_research`
        # `current_node.goal.task_queue.add_new_tasks([Task.construct_from_description(d) for d in task_desc_list.l])`
        # This line is in `Research_Agent.run_research`, not in `Research_Snapshot.insert_tasks`.
        # The `Research_Snapshot.insert_tasks` method is:
        # def insert_tasks(self, task_descriptions: Task_Description_List):
        #   node = self.get_current_node()
        #   node.goal.task_queue.add_new_tasks(task_descriptions)
        # This is problematic because `add_new_tasks` in `Task_Queue` expects `List[Task]`.
        # I will adjust `Research_Snapshot.insert_tasks` to perform the conversion.
        task_list = [Task.construct_from_description(desc) for desc in task_descriptions.l]
        node.goal.task_queue.add_new_tasks(task_list)


    def append_findings(self, finding: str):
        current_node = self.get_current_node()
        current_node.goal.findings.append(finding)
