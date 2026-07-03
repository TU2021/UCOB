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

# --------------------- WebShop --------------------- #
WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment. 
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

Now it's your turn to take one action for the current step.
{thinking_instruction}
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

Now it's your turn to take one action for the current step.
{thinking_instruction}
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_REFLECT_TEMPLATE = """
You are an expert evaluating a WebShop shopping attempt.
Your task is to: {task_description}

You have just completed an attempt at this shopping task. The task was {success} completed.

{reference_trajectory}

Trajectory of the attempt:
{current_trajectory}

If a reference trajectory exists, compare it with the current trajectory.
Given the task outcome, analyze the trajectory to understand:
1. What subtasks were attempted? (search, filter, select, purchase)
2. Which subtasks succeeded vs failed based on the observations?
3. What specific actions or decisions led to this outcome?
4. What are the 1-2 most valuable lessons from this attempt?

Output your evaluation as JSON:

{{
"subtasks": [
{{"name": "search_product", "description": "[describe actual search]", "status": "[completed or incomplete]"}},
{{"name": "apply_filters", "description": "[describe filters used]", "status": "[completed or incomplete]"}},
{{"name": "select_item", "description": "[describe selection]", "status": "[completed or incomplete]"}},
{{"name": "complete_purchase", "description": "[describe purchase]", "status": "[completed or incomplete]"}}
],
"task_success": [true if successfully completed, false if unsuccessfully completed],
"action_lesson": "[key action insight, e.g., 'Precise search with brand+model found exact match' OR 'Generic search missed required features']",
"navigation_lesson": "[navigation insight, e.g., 'Efficient use of filters saved time' OR 'Failed to check additional pages with better options']"
}}

EVALUATION GUIDELINES:
- The task outcome has been provided - use it to set task_success accordingly
- Focus on WHY the attempt had this outcome:
  * If successful: What strategies worked well?
  * If unsuccessful: What went wrong and where?
- Each subtask status must reflect actual trajectory events
- Lessons should explain factors that led to the outcome
- Reference specific elements from trajectory (item IDs, pages, search terms)
- Use null for lessons only if truly not applicable

Output ONLY the JSON evaluation.
"""

WEBSHOP_SKILL_REFLECT_TEMPLATE = """
You are an expert skill writer for the WebShop e-commerce environment.
Your job is to convert task trajectories into one concise, reusable, category-specific shopping skill.

Task category: {task_category}
Task: {task_description}

The attempt being evaluated was {success} completed.

{reference_trajectory}

Trajectory of the attempt being evaluated:
{current_trajectory}

Write a high-level skill in the style of a compact agent memory. The skill must be useful for future WebShop tasks in the same category, not just this exact product.

Think about:
1. What reusable decision principle separates the better trajectory from the failed one, if a reference exists?
2. What should the agent do differently on future {task_category} shopping tasks?
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
- Do not write product-specific item IDs unless they are necessary examples.
- Do not summarize the whole trajectory.
- Do not create a general skill that applies to every category.
- Focus on the category-specific shopping behavior that would prevent this failure or preserve the successful strategy.
"""

WEBSHOP_VLLM_TASK_SKILL_REFLECT_TEMPLATE = """
You are an expert skill writer for the WebShop e-commerce environment.
Your job is to convert shopping trajectories into one concise, reusable memory skill.

Task category: {task_category}
Task: {task_description}

The attempt being evaluated was {success} completed.

{reference_trajectory}

Trajectory of the attempt being evaluated:
{current_trajectory}

Write one high-level shopping skill that would help a future agent solve similar WebShop tasks.
The skill should be abstract enough to transfer across different products, but concrete enough to guide search, comparison, option selection, and purchase decisions.

Think about:
1. What reusable principle separates the better trajectory from the worse one?
2. What should the agent remember for future shopping tasks of this kind?
3. When during a shopping episode should this memory be used?

Output ONLY valid JSON with this schema:

{{
"task_success": false,
"task_category": "{task_category}",
"title": "[short title, 2-6 words]",
"principle": "[one reusable rule for future similar shopping tasks]",
"when_to_apply": "[specific situation in which this skill should be applied]",
"evidence": "[one short note grounding the skill in the trajectory comparison]"
}}

Guidelines:
- Do not summarize the whole trajectory.
- Do not write a skill for only this exact product.
- Avoid exact item ids, prices, product names, and repeated action strings.
- Keep useful shopping concepts such as product category, required attributes, variants, color, size, filters, search terms, price constraints, product pages, and Buy Now when they matter.
- Write one reusable lesson, not a trajectory summary.
"""

WEBSHOP_VLLM_STATE_SKILL_REFLECT_TEMPLATE = """
You are an expert skill writer for the WebShop e-commerce environment.
Your job is to convert a better/worse page-step comparison into one concise state-level memory skill.

Task category: {task_category}
Task: {task_description}

The worse step being evaluated was {success} completed.

{reference_trajectory}

Step being evaluated:
{current_trajectory}

Write one state-level shopping skill explaining the local decision principle that made the better step preferable.
The skill should help a future agent choose a better next action on a similar search result page, product page, or option-selection page.

Think about:
1. What page or option clue mattered?
2. What local action choice was better?
3. When should this memory be used again?

Output ONLY valid JSON with this schema:

{{
"task_success": false,
"task_category": "{task_category}",
"title": "[short title, 2-6 words]",
"principle": "[one reusable local shopping action-choice rule]",
"when_to_apply": "[specific observation condition where this state skill applies]",
"evidence": "[one short note explaining why the better step had higher return than the worse step]"
}}

Guidelines:
- Do not summarize the whole shopping episode.
- Do not copy exact item ids, product names, prices, or option strings.
- Keep the skill abstract, but not vague.
- The memory should sound like a reusable shopping lesson, not a command copied from the trajectory.
"""

WEBSHOP_QUERY_GENERATION_TEMPLATE = """Task: {task_description}
Observation: {initial_observation}

Write a one-sentence search query to find relevant past experiences for this shopping task. Do NOT output an action.
Example: <query>tips for finding a product with the right brand, category, and price</query>

<query>"""

WEBSHOP_RERANK_TEMPLATE = """You are about to attempt a shopping task in the WebShop environment.

Task: {task_description}
Initial Observation: {initial_observation}

Below are {n_candidates} skills retrieved from the skill library. Each is labeled with an ID.

{candidate_experiences}

Rank these experiences from MOST useful to LEAST useful for the current task.
Consider which experience addresses the specific shopping challenge you expect to face.

Output ONLY the ranked IDs as a comma-separated list within <rank> </rank> tags.
For example, if experience 3 is most useful, then 1, then 2: <rank>3,1,2</rank>
"""

WEBSHOP_RERANK_DUMMY_TEMPLATE = """You are about to attempt a shopping task in the WebShop environment.

Task: {task_description}
Initial Observation: {initial_observation}

No past experiences are available for this task. Output <rank>none</rank> to proceed.
"""
