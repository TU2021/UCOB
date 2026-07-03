# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --------------------- ALFWorld --------------------- #
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
{thinking_instruction}
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
{thinking_instruction}
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_REFLECT_TEMPLATE = """
You are an expert evaluating an ALFRED Embodied Environment task attempt.
Your task is to: {task_description}

You have just completed an attempt at this task. The task was {success} completed.

{reference_trajectory}

Trajectory of the attempt:
{current_trajectory}

If a reference trajectory exists, compare it with the current trajectory.
Given the task outcome, analyze the trajectory to understand:
1. What subtasks were attempted? (pick up, navigate, use appliance, place object)
2. Which subtasks succeeded vs failed based on the observations?
3. What specific actions or decisions led to this outcome?
4. What is the most valuable lesson from this attempt?

Output your evaluation as JSON:

{{
"subtasks": [
{{"name": "pick_up_object", "description": "[describe pickup action, e.g., 'Pick up mug from countertop']", "status": "[completed or incomplete]"}},
{{"name": "navigate_to_location", "description": "[describe navigation, e.g., 'Go to microwave 1']", "status": "[completed or incomplete]"}},
{{"name": "use_appliance", "description": "[describe appliance use, e.g., 'Heat mug in microwave']", "status": "[completed or incomplete]"}},
{{"name": "place_object", "description": "[describe placement, e.g., 'Place heated mug in cabinet']", "status": "[completed or incomplete]"}}
],
"task_success": [true if successfully completed task goal, false if failed],
"action_lesson": "[key action insight, e.g., 'Attempted to place mug 1 directly in cabinet 2 without heating - must use microwave 1 first' OR 'Successfully found knife in drawer 3 after checking wrong locations']",
"navigation_lesson": "[spatial insight, e.g., 'Microwave 1 located in kitchen area, not near cabinets' OR 'Multiple sinkbasins exist - must check all for target object']"
}}

EVALUATION GUIDELINES:
- The task outcome has been provided - use it to set task_success accordingly
- Focus on WHY the attempt had this outcome:
  * If successful: What sequence or strategy worked well?
  * If unsuccessful: What step failed or was missed?
- Each subtask status must reflect actual trajectory events
- Lessons should explain factors that led to the outcome
- Reference specific elements from trajectory (object IDs, locations, appliances)
- Use null for lessons only if truly not applicable

Output ONLY the JSON evaluation.
"""

ALFWORLD_SKILL_REFLECT_TEMPLATE = """
You are an expert skill writer for the ALFRED Embodied Environment.
Your job is to convert task trajectories into one concise, reusable, category-specific skill.

Task category: {task_category}
Task: {task_description}

The attempt being evaluated was {success} completed.

{reference_trajectory}

Trajectory of the attempt being evaluated:
{current_trajectory}

Write a high-level skill in the style of a compact agent memory. The skill must be useful for future tasks in the same category, not just this exact scene.

Think about:
1. What reusable decision principle separates the better trajectory from the failed one, if a reference exists?
2. What should the agent do differently on future {task_category} tasks?
3. When exactly should this skill be applied during an episode?

Output ONLY valid JSON with this schema:

{{
"task_success": [true if the evaluated attempt succeeded, false otherwise],
"task_category": "{task_category}",
"title": "[short imperative title, 2-6 words]",
"principle": "[one concise reusable rule for future tasks in this category]",
"when_to_apply": "[specific situation in which the agent should apply this skill]",
"evidence": "[one short note grounding the skill in the trajectory comparison]"
}}

Guidelines:
- Do not write scene-specific object IDs unless they are necessary examples.
- Do not summarize the whole trajectory.
- Do not create a general skill that applies to every category.
- Focus on the category-specific behavior that would prevent this failure or preserve the successful strategy.
"""

ALFWORLD_VLLM_TASK_SKILL_REFLECT_TEMPLATE = """
You are an expert skill writer for the ALFRED Embodied Environment.
Your job is to convert task trajectories into one concise, reusable memory skill.

Task category: {task_category}
Task: {task_description}

The attempt being evaluated was {success} completed.

{reference_trajectory}

Trajectory of the attempt being evaluated:
{current_trajectory}

Write one high-level skill that would help a future agent solve similar ALFWorld tasks.
The skill should be abstract enough to transfer across different objects and layouts, but concrete enough to guide behavior in this environment.

Think about:
1. What reusable principle separates the better trajectory from the worse one?
2. What should the agent remember for future tasks of this kind?
3. When during an episode should this memory be used?

Output ONLY valid JSON with this schema:

{{
"task_success": false,
"task_category": "{task_category}",
"title": "[short title, 2-6 words]",
"principle": "[one reusable rule for future similar tasks]",
"when_to_apply": "[specific situation in which this skill should be applied]",
"evidence": "[one short note grounding the skill in the trajectory comparison]"
}}

Guidelines:
- Do not summarize the whole trajectory.
- Do not write a skill for only this exact scene.
- Avoid exact object ids, location ids, and repeated action strings.
- Keep useful task concepts such as cooling, heating, cleaning, placing, searching, containers, receptacles, appliances, or light sources when they matter.
- Write one reusable lesson, not a trajectory summary.
"""

ALFWORLD_VLLM_STATE_SKILL_REFLECT_TEMPLATE = """
You are an expert skill writer for the ALFRED Embodied Environment.
Your job is to convert a better/worse step comparison into one concise state-level memory skill.

Task category: {task_category}
Task: {task_description}

The worse step being evaluated was {success} completed.

{reference_trajectory}

Step being evaluated:
{current_trajectory}

Write one state-level skill explaining the local decision principle that made the better step preferable.
The skill should help a future agent choose a better next action in a similar observation.

Think about:
1. What observation clue mattered?
2. What local action choice was better?
3. When should this memory be used again?

Output ONLY valid JSON with this schema:

{{
"task_success": false,
"task_category": "{task_category}",
"title": "[short title, 2-6 words]",
"principle": "[one reusable local action-choice rule]",
"when_to_apply": "[specific observation condition where this state skill applies]",
"evidence": "[one short note explaining why the better step had higher return than the worse step]"
}}

Guidelines:
- Do not summarize the whole episode.
- Do not copy exact object ids or location ids.
- Keep the skill abstract, but not vague.
- The memory should sound like a reusable agent lesson, not a command copied from the trajectory.
"""

ALFWORLD_QUERY_GENERATION_TEMPLATE = """Task: {task_description}
Observation: {initial_observation}

Write a one-sentence search query to find relevant past experiences for this task. Do NOT output an action.
Example: <query>tips for opening containers before picking up objects</query>

<query>"""

ALFWORLD_RERANK_TEMPLATE = """You are about to attempt a task in the ALFRED Embodied Environment.

Task: {task_description}
Initial Observation: {initial_observation}

Below are {n_candidates} skills retrieved from the skill library. Each is labeled with an ID.

{candidate_experiences}

Rank these experiences from MOST useful to LEAST useful for the current task.
Consider which experience addresses the specific task challenge you expect to face.

Output ONLY the ranked IDs as a comma-separated list within <rank> </rank> tags.
For example, if experience 3 is most useful, then 1, then 2: <rank>3,1,2</rank>
"""

ALFWORLD_RERANK_DUMMY_TEMPLATE = """You are about to attempt a task in the ALFRED Embodied Environment.

Task: {task_description}
Initial Observation: {initial_observation}

No past experiences are available for this task. Output <rank>none</rank> to proceed.
"""
