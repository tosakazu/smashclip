# Claude-as-Judge Prompt for Task D Rationale Evaluation

This prompt is sent to Claude to evaluate generated rationales against ground-truth captions.

## Prompt Template

```
You are evaluating the quality of a generated explanation for a Super Smash Bros. Ultimate KO clip.

Ground truth explanation: "{gt_caption}"
Generated explanation: "{generated}"

Rate the generated explanation on three criteria (1-5 each):
1. Relevance: Does it address the same aspects of the clip?
2. Specificity: Does it mention specific moves, techniques, or situations?
3. Accuracy: Is the assessment consistent with the ground truth?

Respond with only three integers separated by commas. Example: 4,3,5
```

## Evaluation Axes

### Relevance (1-5)

Does the rationale relate to the clip's actual content?

| Score | Meaning |
|-------|---------|
| 1 | Completely off-topic or generic boilerplate |
| 2 | Vaguely related but misses the main aspect of the clip |
| 3 | Addresses some of the same aspects as the ground truth |
| 4 | Covers most of the same aspects |
| 5 | Addresses the same aspects with comparable depth |

### Specificity (1-5)

Does the rationale mention concrete details like move names, character abilities, or combo sequences?

| Score | Meaning |
|-------|---------|
| 1 | No specific details; entirely generic |
| 2 | One vague reference to game mechanics |
| 3 | Some specific details (e.g., mentions a move name or technique) |
| 4 | Multiple concrete details that match the clip |
| 5 | Rich in specific, correct details about moves, combos, and context |

### Accuracy (1-5)

Are the descriptions of characters, moves, and situations factually correct within SSBU?

| Score | Meaning |
|-------|---------|
| 1 | Major factual errors (wrong characters, impossible moves) |
| 2 | Several inaccuracies in game-specific details |
| 3 | Mostly correct with minor errors |
| 4 | Accurate with at most one small mistake |
| 5 | Fully accurate; all details match the ground truth and game mechanics |

## Parsing

The response is parsed as three comma-separated integers. If parsing fails, the clip is marked as a parse error and excluded from the aggregate.

Aggregate scores are reported as the mean across all evaluated clips for each axis.
