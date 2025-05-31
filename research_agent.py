from enum import Enum
from typing import Optional, List, Dict, Any, Type, TypeVar

from llama_index.core import PromptTemplate
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.bridge.pydantic import Field # BaseModel, ValidationError removed
from pydantic import create_model # BaseModel from pydantic is not the one used by data_models typically.

# Imports from our new modules
from data_models import ReasonOnly, Goal_Description, SplitGoals, Task_Description_List # GoalDecision, Task_Description removed
from task_management import Task # Task_Queue removed
from goal_management import Research_Snapshot # Goal_Info, Snapshot_Node removed

__DEBUG__ = True

E = TypeVar('E', bound=Enum)


class Research_Agent:
    # Nested Enums used by the agent's decision logic
    class SplitDecision(Enum):
        ENOUGH = "enough"
        NO_SPLIT = "no_split"
        SPLIT = "split"

    # Removed nested Research_Agent.ReasonOnly as it's unused and data_models.ReasonOnly is used.

    class PrepDecision(Enum):
        NEED_PREP = "need_prep"
        READY = "ready_to_split"

    def __init__(self, research_topic: str):
        # Initialize snapshot with the main goal
        self.snapshot = Research_Snapshot.initialize_from_usr_request(Goal_Description(description=research_topic))
        # TODO: Securely manage API keys
        self.plan_llm = GoogleGenAI(model="gemini-2.0-flash",
                                    api_key="AIzaSyC-CRUyOv_MgrKis-8sm-_w8eugKVI_S_k", # Placeholder
                                    temperature=0.7)
        self.worker_llm = GoogleGenAI(model="gemini-2.0-flash",
                                      api_key="AIzaSyC-CRUyOv_MgrKis-8sm-_w8eugKVI_S_k", # Placeholder
                                      temperature=0.1)
        # self.act_llm is used in _summarize, but not initialized. This was an error in the original code.
        # Assuming it should be similar to plan_llm or worker_llm. For now, let's use plan_llm.
        self.act_llm = self.plan_llm


    def _decide_enum(
            self,
            enum_type: Type[E], # E is a TypeVar bound to Enum
            prompt_str: str,
            prompt_vars: Dict[str, Any]
    ) -> E: # Return type is an instance of the Enum pass in enum_type
        """
        Build a one‐off Pydantic model whose `decision` field is enum_type,
        call structured_predict, and return the parsed enum member.
        The base model for this dynamic model is data_models.ReasonOnly.
        """
        DecisionModel = create_model(
            'DecisionModel', # Dynamically created model name
            __base__=(ReasonOnly,), # Base model from data_models.py
            decision=(enum_type, Field( # The actual decision field, type is the passed enum_type
                ..., # Ellipsis means this field is required
                description=(
                        "Return exactly one of: "
                        + ", ".join(f"'{m.name}'" for m in enum_type) # Dynamic description
                )
            ))
        )

        prompt = PromptTemplate(prompt_str)
        # structured_predict will return an instance of DecisionModel
        ans: DecisionModel = self.plan_llm.structured_predict(
            DecisionModel, # The model to parse the output into
            prompt,
            **prompt_vars
        )

        if __DEBUG__:
            print(f"Reason for decision: {ans.reason}")
        return ans.decision # Return the 'decision' field from the parsed model

    def _decide_need_prep(
            self,
            current_goal_desc: str, # Description of the current goal
            previous_findings: Optional[str] = None # Optional previous findings
    ) -> PrepDecision: # Returns a member of the PrepDecision enum
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
        vars_dict = {
            "current_goal_desc": current_goal_desc,
            "previous_findings_block": previous_findings or "(none)"
        }
        # Call _decide_enum with PrepDecision enum type
        return self._decide_enum(Research_Agent.PrepDecision, prompt, vars_dict)

    def _decide_split(
            self,
            current_goal_desc: str, # Description of the current goal
            subgoal_summaries: Optional[List[str]] = None # Optional list of sub-goal summaries
    ) -> SplitDecision: # Returns a member of the SplitDecision enum
        subgoals_block = (
            "\n".join(f"- {s}" for s in subgoal_summaries) # Format list of summaries
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
        vars_dict = {
            "current_goal_desc": current_goal_desc,
            "subgoals_block": subgoals_block
        }
        # Call _decide_enum with SplitDecision enum type
        return self._decide_enum(Research_Agent.SplitDecision, prompt, vars_dict)

    def _split_goal(
            self,
            current_goal_desc: Goal_Description, # Current goal as Goal_Description object
            previous_findings: str # Previous findings as a string
    ) -> List[Goal_Description]: # Returns a list of Goal_Description objects
        """
        Decompose `current_goal_desc` into a sequence of sub-goals (as Goal_Description objects),
        using `previous_findings` as context.
        """
        prompt_template = PromptTemplate( # Corrected variable name
            "You are a Deep Research Agent. Your task is to decompose the main research goal\n"
            "into a logically ordered sequence of INDEPENDENT sub-goals. Each sub-goal must be:\n"
            "   + Broad enough to stand as an independent research objective"
            "   + Necessary for achieving the overall goal "
            "Main goal:\n"
            "{goal_desc}\n\n"
            "Previous findings (if any):\n"
            "{previous_findings}\n\n"
        )

        # structured_predict will return an instance of SplitGoals
        result: SplitGoals = self.plan_llm.structured_predict(
            SplitGoals, # The Pydantic model for the expected output (list of subgoals)
            prompt_template, # Use the PromptTemplate instance
            goal_desc=current_goal_desc.description,
            previous_findings=previous_findings or "(none)"
        )

        return result.subgoals

    def _summarize(self, info: str):
        # This method used self.act_llm which was not initialized.
        # It's now initialized to self.plan_llm in __init__.
        ans = self.act_llm.predict( # Now self.act_llm is defined
            PromptTemplate(
                "Summarize the content. Make sure your summarize is linked to the goal"
                "Content: {info}"
            ),
            info=info
        )
        # global __DEBUG__ # __DEBUG__ is a global, no need to declare global to read
        if __DEBUG__:
            print(f"Summary: {ans}") # Added a label for clarity

        return ans

    def _plan_tasks(self, background: str, target: str) -> Task_Description_List:
        prompt_template = PromptTemplate( # Corrected variable name
            "You are a Deep Research Planner Agent. Your goal is to provide a list of task descriptions for another search agent.\n"
            "These tasks has to be independent and ordered and should be able to achieve the following objective:\n"
            "{target}\n\n"
            "Use this background information to inform your planning:\n"
            "{background}\n\n"
            ""
        )
        # structured_predict will return an instance of Task_Description_List
        result: Task_Description_List = self.plan_llm.structured_predict(
            Task_Description_List, # The Pydantic model for the expected output
            prompt_template, # Use the PromptTemplate instance
            target=target,
            background=background
        )

        return result

    def _execute_pending_tasks(self) -> List[str]: # Added return type hint
        def _llm_worker_stub(task: Task) -> Task: # Added param and return type hints
            # This is a stub for actual task execution logic
            ans = self.worker_llm.predict(
                prompt=PromptTemplate(f"You are a search engine. return result based on {task.description.action}" # Access action field
                                      f"be concise")
            )
            task.status = "SUCCEEDED" # Mark task as succeeded
            task.result = ans # Store result in task
            return task

        current_node = self.snapshot.get_current_node()
        findings: List[str] = [] # Initialize findings list
        # Iterate through the task queue of the current goal node
        for pending_task in current_node.goal.task_queue: # Task_Queue is iterable
            _task = _llm_worker_stub(pending_task) # Execute task
            if _task.result: # Ensure result is not None
                findings.append(_task.result)
            current_node.goal.task_queue.commit_task(_task) # Commit task updates status
        return findings

    def run_research(self):
        current_node = self.snapshot.get_current_node()
        if not current_node:
            print("Research ended: No current node to process.")
            return

        sub_node_summary: List[str] = []
        for sub_node_idx in current_node.children:
            # Ensure sub_node_idx is not None if that's possible from self.nodes[idx].children
            if sub_node_idx is None: continue # Should not happen with current logic List[Optional[int]]

            chld_node = self.snapshot.nodes[sub_node_idx]
            # Original code: chld_node.goal.status != "SUCCEED" (Typo, should be SUCCEEDED)
            if chld_node.goal.status != "SUCCEEDED":
                # If any child goal is not yet succeeded, we might not be ready to summarize or decide on parent.
                # This part of logic might need refinement: what if a child is still RUNNING or QUEUED?
                # For now, adhering to original logic of only proceeding if children are "SUCCEEDED" (fixed typo)
                print(f"Child goal {chld_node.goal.description.description} not yet SUCCEEDED. Status: {chld_node.goal.status}")
                # Depending on desired behavior, may return False or handle differently
                return False # Or continue to check other children / parent status
            sub_node_summary.append(chld_node.goal.summary or "") # Use summary if available

        # Decide how to proceed with the current_node's goal
        choice = self._decide_split(current_goal_desc=current_node.goal.description.description, # Access actual description string
                                    subgoal_summaries=sub_node_summary,
                                    )

        if choice is Research_Agent.SplitDecision.ENOUGH:
            # Goal is considered complete. Potentially summarize findings.
            current_node.goal.status = "SUCCEEDED"
            current_node.goal.summary = self._summarize(str(current_node.goal.findings))
            print(f"Goal '{current_node.goal.description.description}' marked as ENOUGH (SUCCEEDED).")
            # Potentially explore parent or sibling, or end if root.
            parent_idx = self.snapshot.get_parent_idx()
            if parent_idx is not None:
                self.snapshot.active_cursor = parent_idx
                self.run_research() # Continue with parent
            else:
                print("Root goal marked as ENOUGH. Research complete.")

        elif choice is Research_Agent.SplitDecision.NO_SPLIT:
            # Goal does not need splitting, but requires work (tasks).
            # This assumes 'NO_SPLIT' means 'execute tasks for this goal'.
            print(f"Goal '{current_node.goal.description.description}' decided as NO_SPLIT. Planning and executing tasks.")
            # Plan tasks to achieve the current goal if no tasks exist or if they need replanning.
            if not current_node.goal.task_queue.unfinished_queue and not current_node.goal.task_queue.current_task :
                 task_desc_list = self._plan_tasks(
                    background=f"Previous findings: {str(current_node.goal.findings)}",
                    target=f"Achieve the goal: {current_node.goal.description.description}"
                )
                 # The add_new_tasks in Task_Queue expects List[Task]
                 # The _plan_tasks returns Task_Description_List
                 # Research_Snapshot.insert_tasks handles this conversion.
                 # However, here we are adding directly to task_queue of a goal.
                 tasks_to_add = [Task.construct_from_description(d) for d in task_desc_list.l]
                 current_node.goal.task_queue.add_new_tasks(tasks_to_add)

            new_findings = self._execute_pending_tasks()
            current_node.goal.findings.extend(new_findings)
            # After executing tasks, re-evaluate. For instance, mark as ENOUGH or re-split.
            # This might lead to a loop or a state machine logic here.
            # For now, let's assume executing tasks might lead to 'ENOUGH' in a subsequent run_research call.
            current_node.goal.status = "SUCCEEDED" # Or some other status like 'PENDING_REVIEW'
            current_node.goal.summary = self._summarize(str(current_node.goal.findings))
            print(f"Tasks executed for '{current_node.goal.description.description}'. Goal marked SUCCEEDED for now.")
            # Potentially move to parent or explore next if this was part of a sequence.
            # self.snapshot.explore() # This would move to next sub-goal if any, or parent.

        elif choice is Research_Agent.SplitDecision.SPLIT:
            print(f"Goal '{current_node.goal.description.description}' decided as SPLIT.")
            previous_findings_str = str(current_node.goal.findings)
            max_retries = 10 # Max retries for prep-work before splitting
            trial_time = 0
            while trial_time < max_retries:
                # Decide if prep work (more info gathering) is needed before splitting
                prep_decision = self._decide_need_prep(current_goal_desc=current_node.goal.description.description,
                                                       previous_findings=previous_findings_str)
                if prep_decision is Research_Agent.PrepDecision.NEED_PREP:
                    print("Need prep work before splitting...")
                    task_desc_list = self._plan_tasks(
                        background=f"Goal: {current_node.goal.description.description}\nPrevious Findings: {previous_findings_str}\n",
                        target="Get enough information to split the Goal: landscape survey "
                               "(For example, what are the aspects that has been discussed,"
                               " what materials should look into...)"
                               "Tasks should be a bit concrete and search-friendly. "
                               "you may make the list longer"
                    )
                    # Add tasks to the current node's goal task queue
                    tasks_to_add = [Task.construct_from_description(d) for d in task_desc_list.l]
                    current_node.goal.task_queue.add_new_tasks(tasks_to_add)

                    new_findings = self._execute_pending_tasks() # Execute these prep tasks
                    current_node.goal.findings.extend(new_findings)
                    previous_findings_str = str(current_node.goal.findings) # Update findings
                    trial_time += 1
                else: # Ready to split
                    print("Ready to split the goal.")
                    break

            if trial_time == max_retries:
                print("Max retries for prep work reached. Proceeding to split with current info.")

            # Split the current goal into sub-goals
            sub_goal_descs = self._split_goal(current_goal_desc=current_node.goal.description,
                                              previous_findings=previous_findings_str)
            if not sub_goal_descs:
                print("Splitting returned no sub-goals. Marking current goal as NO_SPLIT temporarily.")
                # Treat as NO_SPLIT, try to execute tasks for it or mark as stuck
                # This is a fallback, ideally split should yield subgoals or agent should decide differently
                # For now, let's try to run tasks for it (similar to NO_SPLIT path)
                self.run_research() # Re-run, hoping it resolves to NO_SPLIT or ENOUGH
                return

            for sub_goal_desc in sub_goal_descs:
                # Create a new snapshot node for each sub-goal and insert it
                new_node = Research_Snapshot.create_goal_node(sub_goal_desc)
                self.snapshot.insert_node(new_node) # insert_node links it to current_node as parent

            # After adding sub-goals, explore the first one
            explored_node = self.snapshot.explore()
            if explored_node and explored_node is not current_node: # if explore returned a child
                 self.run_research() # recurse on the new active node (child)
            elif explored_node is current_node: # explore returned the same node (parent, children done)
                 current_node.goal.status = "SUCCEEDED" # Mark parent as succeeded if children are done
                 self.run_research() # re-evaluate parent
            else: # explore returned None (e.g. root and children done)
                 print("Exploration returned None after splitting. Research might be complete or stuck.")
        else:
            raise ValueError(f"Unrecognized choice: {choice.name if isinstance(choice, Enum) else choice}")

# Example execution
if __name__ == "__main__": # Guard for script execution
    # TODO: Replace with actual research topic from user or config
    r = Research_Agent(research_topic="the creation of US dollar hegemony")
    r.run_research()
