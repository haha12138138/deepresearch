from enum import Enum
from typing import List

from pydantic import BaseModel, Field

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


class Goal_Description(BaseModel):
    description: str = Field(..., description="describe the goal clearly and accurately, "
                                              "and includes the intention and expectation of this research goal")


class SplitGoals(BaseModel):
    subgoals: List[Goal_Description] = Field(
        ...,
        description="An ordered list of sub-goals, each as a Goal_Description object."
    )
