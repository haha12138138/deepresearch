from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, List, Literal, Dict, Any, Type, TypeVar

from llama_index.core import PromptTemplate
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.bridge.pydantic import BaseModel, Field, ValidationError
from collections import deque

__DEBUG__ = True

from pydantic import create_model


# 1) Define your four cases
class GoalDecision(Enum):
    ENOUGH = "enough"  # finished
    NO_SPLIT = "no_split"  # simple enough, no decomposition
    SPLIT = "split"  # needs decomposition
    SPLIT_NEED_HELP = "split_need_help"  # needs prep work before splitting


# 2) Simple base for the reason field
class ReasonOnly(BaseModel):
    reason: str = Field(
        ...,
        description="Briefly describe your reason for choosing this case (1 sentence)."
    )


class Task_Description(BaseModel):
    action: str = Field("describe the task, can include some guidance about how to achieve the goal")


class Task_Description_List(BaseModel):
    l: List[Task_Description] = Field(..., description="make sure that the task is ordered:"
                                                       "the task with smaller idx in the list shouldn't depend on the "
                                                       "task with larger index")


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


class Goal_Description(BaseModel):
    description: str = Field(..., description="describe the goal clearly and accurately, "
                                              "and includes the intention and expectation of this research goal")


class SplitGoals(BaseModel):
    subgoals: List[Goal_Description] = Field(
        ...,
        description="An ordered list of sub-goals, each as a Goal_Description object."
    )


class Goal_Info:
    def __init__(self, description: Goal_Description):
        self.description = description
        self.status: Literal['QUEUED', 'RUNNING', 'SUCCEEDED', 'ERROR'] = "QUEUED"
        self.findings: [str] = []
        self.summary: str = ""
        self.task_queue: Task_Queue = Task_Queue()

    def __str__(self):
        return f"Description: {self.description}, findings {self.findings}"

    @classmethod
    def construct_from_description(cls, description: Goal_Description):
        new_goal = cls(description)
        return new_goal


class Snapshot_Node:
    def __init__(self, goal_info):
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
    def create_goal_node(cls, goal_description: Goal_Description) -> Snapshot_Node:
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
        self.nodes: List[Snapshot_Node] = []
        self._idx_tracker = 0

    def insert_node(self, node: Snapshot_Node):
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

    def get_current_node(self) -> Snapshot_Node:
        if self.active_cursor is not None:
            return self.nodes[self.active_cursor]
        else:
            return None

    def explore(self) -> Optional[Snapshot_Node]:
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
        node.goal.task_queue.add_new_tasks(task_descriptions)

    def append_findings(self, finding: str):
        current_node = self.get_current_node()
        current_node.goal.findings.append(finding)


E = TypeVar('E', bound=Enum)


class Research_Agent:
    class SplitDecision(Enum):
        ENOUGH = "enough"
        NO_SPLIT = "no_split"
        SPLIT = "split"

    class ReasonOnly(BaseModel):
        reason: str = Field(
            ...,
            description="Briefly describe your reason for choosing this case (1 sentence)."
        )

    class PrepDecision(Enum):
        NEED_PREP = "need_prep"
        READY = "ready_to_split"

    def __init__(self, research_topic: str):
        self.snapshot = Research_Snapshot.initialize_from_usr_request(Goal_Description(description=research_topic))
        self.plan_llm = GoogleGenAI(model="gemini-2.0-flash",
                                    api_key="AIzaSyC-CRUyOv_MgrKis-8sm-_w8eugKVI_S_k",
                                    temperature=0.7)
        self.worker_llm = GoogleGenAI(model="gemini-2.0-flash",
                                      api_key="AIzaSyC-CRUyOv_MgrKis-8sm-_w8eugKVI_S_k",
                                      temperature=0.1)

    def _decide_enum(
            self,
            enum_type: Type[E],
            prompt_str: str,
            prompt_vars: Dict[str, Any]
    ) -> E:
        """
        Build a one‐off Pydantic model whose `decision` field is enum_type,
        call structured_predict, and return the parsed enum member.
        """
        DecisionModel = create_model(
            'DecisionModel',
            __base__=(ReasonOnly,),
            decision=(enum_type, Field(
                ...,
                description=(
                        "Return exactly one of: "
                        + ", ".join(f"'{m.name}'" for m in enum_type)
                )
            ))
        )

        prompt = PromptTemplate(prompt_str)
        ans: DecisionModel = self.plan_llm.structured_predict(
            DecisionModel,
            prompt,
            **prompt_vars
        )

        if __DEBUG__:
            print(ans.reason)
        return ans.decision

    def _decide_need_prep(
            self,
            current_goal_desc: str,
            previous_findings: Optional[str] = None
    ) -> PrepDecision:
        prompt = (
            "You’re deciding whether you have enough background to decompose the current goal.\n\n"
            "Current goal:\n"
            "{current_goal_desc}\n\n"
            "Previous findings (if any):\n"
            "{previous_findings_block}\n\n"
            "Return exactly one of:\n"
            "  • need_prep       – the goal is too complex; first gather more info.\n"
            "  • ready_to_split  – you have enough context; proceed to splitting.\n"
        )
        vars = {
            "current_goal_desc": current_goal_desc,
            "previous_findings_block": previous_findings or "(none)"
        }
        return self._decide_enum(Research_Agent.PrepDecision, prompt, vars)

    def _decide_split(
            self,
            current_goal_desc: str,
            subgoal_summaries: Optional[List[str]] = None
    ) -> SplitDecision:
        subgoals_block = (
            "\n".join(f"- {s}" for s in subgoal_summaries)
            if subgoal_summaries else
            "(no sub-goals yet)"
        )
        prompt = (
            "You’re deciding how to handle the current goal now that you’re ready.\n\n"
            "Current goal:\n"
            "{current_goal_desc}\n\n"
            "Sub-goal summaries (if any):\n"
            "{subgoals_block}\n\n"
            "Return exactly one of:\n"
            "  • enough    – we’re finished with this goal.\n"
            "  • no_split  – not done but simple enough; no decomposition needed.\n"
            "  • split     – it’s complex/broad; decompose into sub-goals.\n"
        )
        vars = {
            "current_goal_desc": current_goal_desc,
            "subgoals_block": subgoals_block
        }
        return self._decide_enum(Research_Agent.SplitDecision, prompt, vars)

    def _split_goal(
            self,
            current_goal_desc: Goal_Description,
            previous_findings: str
    ) -> List[Goal_Description]:
        """
        Decompose `current_goal_desc` into a sequence of sub-goals (as Goal_Description objects),
        using `previous_findings` as context.
        """
        prompt = PromptTemplate(
            "You are a Deep Research Agent. Your task is to decompose the main research goal\n"
            "into a logically ordered sequence of INDEPENDENT sub-goals. Each sub-goal must be:\n"
            "   + Broad enough to stand as an independent research objective"
            "   + Necessary for achieving the overall goal "
            "Main goal:\n"
            "{goal_desc}\n\n"
            "Previous findings (if any):\n"
            "{previous_findings}\n\n"
        )

        result: SplitGoals = self.plan_llm.structured_predict(
            SplitGoals,
            prompt,
            goal_desc=current_goal_desc.description,
            previous_findings=previous_findings or "(none)"
        )

        return result.subgoals

    def _summarize(self, info: str):
        ans = self.act_llm.predict(
            PromptTemplate(
                "Summarize the content. Make sure your summarize is linked to the goal"
                "Content: {info}"
            ),
            info=info
        )
        global __DEBUG__
        if __DEBUG__:
            print(ans)

        return ans

    def _plan_tasks(self, background: str, target: str) -> Task_Description_List:
        prompt = PromptTemplate(
            "You are a Deep Research Planner Agent. Your goal is to provide a list of task descriptions for another search agent.\n"
            "These tasks has to be independent and ordered and should be able to achieve the following objective:\n"
            "{target}\n\n"
            "Use this background information to inform your planning:\n"
            "{background}\n\n"
            ""
        )
        result: Task_Description_List = self.plan_llm.structured_predict(
            Task_Description_List,
            prompt,
            target=target,
            background=background
        )

        return result

    def _execute_pending_tasks(self):
        def _llm_worker_stub(task: Task):
            ans = self.worker_llm.predict(
                prompt=PromptTemplate(f"You are a search engine. return result based on {task.description}"
                                      f"be concise")
            )
            task.status = "SUCCEEDED"
            task.result = ans
            return task

        current_node = self.snapshot.get_current_node()
        findings = []
        for pending_task in current_node.goal.task_queue:
            _task = _llm_worker_stub(pending_task)
            findings.append(_task.result)
            current_node.goal.task_queue.commit_task(_task)
        return findings

    def run_research(self):
        current_node = self.snapshot.get_current_node()
        sub_node_summary = []
        for sub_node_idx in current_node.children:
            chld_node = self.snapshot.nodes[sub_node_idx]
            if chld_node.goal.status != "SUCCEED":
                return False
            sub_node_summary += [chld_node.goal.summary]

        choice = self._decide_split(current_goal_desc=current_node.goal.description,
                                    subgoal_summaries=sub_node_summary,
                                    )
        if choice is self.SplitDecision.ENOUGH:
            ...
        elif choice is self.SplitDecision.NO_SPLIT:
            ...
        elif choice is self.SplitDecision.SPLIT:
            previous_findings = []
            max_retries = 10
            trial_time = 0
            while trial_time < max_retries:
                decision = self._decide_need_prep(current_goal_desc=current_node.goal.description,
                                                  previous_findings=str(previous_findings))
                if decision is self.PrepDecision.NEED_PREP:
                    task_desc_list = self._plan_tasks(background=f"Goal: {current_node.goal.description}\n"
                                                                 f"Previous Findings: {str(previous_findings)}\n",
                                                      target="Get enough information to split the Goal: landscape survey "
                                                             "(For example, what are the aspects that has been discussed,"
                                                             " what materials should look into...)"
                                                             "Tasks should be a bit concrete and search-friendly. "
                                                             "you may make the list longer")
                    current_node.goal.task_queue.add_new_tasks(
                        [Task.construct_from_description(d) for d in task_desc_list.l])
                    previous_findings += self._execute_pending_tasks()
                    trial_time += 1
                else:
                    break
            sub_goal_list = self._split_goal(current_goal_desc=current_node.goal.description,
                                             previous_findings=str(previous_findings))
            for sub_goal_desc in sub_goal_list:
                self.snapshot.insert_node(Research_Snapshot.create_goal_node(sub_goal_desc))

            self.snapshot.explore()

        else:
            raise ValueError(f"Unrecognized choice: {choice.name}")


r = Research_Agent(research_topic="the creation of US dollar hegemony")
r.run_research()
