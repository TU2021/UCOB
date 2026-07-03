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

SEARCH_TEMPLATE_NO_HIS = """
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Now it's your turn to respond for the current step.
{thinking_instruction}
After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If you find you lack some knowledge, you can call a search engine to get more external information using format: <search> your query </search>.
(2) If you have enough knowledge to answer the question confidently, provide your final answer within <answer> </answer> tags, without detailed illustrations. For example, <answer>Beijing</answer>.
"""

SEARCH_TEMPLATE = """
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below is the interaction history where <search> </search> wrapped your past search queries and <information> </information> wrapped the corresponding search results returned by the external search engine. History:
{memory_context}

Now it's your turn to respond for the current step.
{thinking_instruction}
After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If you find you lack some knowledge, you can call a search engine to get more external information using format: <search> your query </search>.
(2) If you have enough knowledge to answer the question confidently, provide your final answer within <answer> </answer> tags, without detailed illustrations. For example, <answer>Beijing</answer>.
"""

SEARCH_REFLECT_TEMPLATE = """
You are an expert evaluating a search-based question answering attempt.
Question: {task_description}

The attempt being evaluated was {success} completed.

{reference_trajectory}

Trajectory of the attempt being evaluated:
{current_trajectory}

Analyze whether the trajectory made progress on useful search subtasks: decomposing the question, choosing targeted queries, reading evidence, refining weak results, and answering only when evidence is sufficient.

Output ONLY valid JSON with this schema:

{{
"task_success": [true if the evaluated attempt succeeded, false otherwise],
"action_lesson": "[one concrete lesson about search query or answer timing]",
"navigation_lesson": "[one concrete lesson about evidence gathering, decomposition, or refinement]",
"trajectory_value": "[integer from 0 to 6 estimating completed search subtasks]"
}}
"""

SEARCH_SKILL_REFLECT_TEMPLATE = """
You are an expert skill writer for a search-based question answering agent.
Your job is to convert search trajectories into one concise, reusable, category-specific skill.

Task category: {task_category}
Question: {task_description}

The attempt being evaluated was {success} completed.

{reference_trajectory}

Trajectory of the attempt being evaluated:
{current_trajectory}

Write a high-level skill in the style of a compact agent memory. The skill must be useful for future questions in the same category, not just this exact question.

Think about:
1. What reusable decision principle separates the better trajectory from the failed one, if a reference exists?
2. How should the agent choose, refine, or stop search queries on future {task_category} questions?
3. When exactly should this skill be applied during a search episode?

Output ONLY valid JSON with this schema:

{{
"task_success": [true if the evaluated attempt succeeded, false otherwise],
"task_category": "{task_category}",
"title": "[short imperative title, 2-6 words]",
"principle": "[one concise reusable search rule]",
"when_to_apply": "[specific question or search-history situation in which the agent should apply this skill]",
"evidence": "[one short note grounding the skill in the trajectory comparison]"
}}

Guidelines:
- Do not summarize the whole trajectory.
- Do not copy a full query sequence.
- Do not write a skill that applies to every possible question.
- Focus on reusable behavior: decomposing multi-hop questions, crafting precise queries, refining weak evidence, comparing facts, resolving ambiguity, or stopping when evidence is sufficient.
"""

SEARCH_VLLM_TASK_SKILL_REFLECT_TEMPLATE = """
You are an expert skill writer for a search-based question answering agent.
Your job is to convert search trajectories into one concise, reusable memory skill.

Task category: {task_category}
Question: {task_description}

The attempt being evaluated was {success} completed.

{reference_trajectory}

Trajectory of the attempt being evaluated:
{current_trajectory}

Write one high-level search skill that would help a future agent answer similar questions.
The skill should be abstract enough to transfer across questions, but concrete enough to guide search behavior, evidence use, and final-answer timing.

Think about:
1. What reusable principle separates the better trajectory from the worse one?
2. What should the agent remember before issuing the next query or final answer?
3. When during a search episode should this memory be used?

Output ONLY valid JSON with this schema:

{{
"task_success": false,
"task_category": "{task_category}",
"title": "[short title, 2-6 words]",
"principle": "[one reusable search rule]",
"when_to_apply": "[specific question or evidence condition where this skill applies]",
"evidence": "[one short note grounding the skill in the trajectory comparison]"
}}

Guidelines:
- Do not summarize the whole trajectory.
- Do not write a skill for only this exact question.
- Avoid copying exact retrieved snippets or a full query sequence.
- Keep useful search concepts such as decomposition, exact entity names, attribute lookup, query refinement, source evidence, comparison, ambiguity, and answer timing when they matter.
- Write one reusable lesson, not a trajectory summary.
"""

SEARCH_VLLM_STATE_SKILL_REFLECT_TEMPLATE = """
You are an expert skill writer for a search-based question answering agent.
Your job is to convert a better/worse step comparison into one concise state-level memory skill.

Task category: {task_category}
Question: {task_description}

The worse step being evaluated was {success} completed.

{reference_trajectory}

Step being evaluated:
{current_trajectory}

Write one state-level search skill explaining the local decision principle that made the better step preferable.
The skill should help a future agent choose a better next search query or final answer in a similar question/history state.

Think about:
1. What clue in the question or search history mattered?
2. Was it better to search, refine the query, compare evidence, or answer?
3. When should this memory be used again?

Output ONLY valid JSON with this schema:

{{
"task_success": false,
"task_category": "{task_category}",
"title": "[short title, 2-6 words]",
"principle": "[one reusable local search-decision rule]",
"when_to_apply": "[specific question/history condition where this state skill applies]",
"evidence": "[one short note explaining why the better step had higher return than the worse step]"
}}

Guidelines:
- Do not summarize the whole episode.
- Do not copy exact retrieved snippets or a full query sequence.
- Keep the skill abstract, but not vague.
- The memory should sound like a reusable search lesson, not a command copied from the trajectory.
"""
