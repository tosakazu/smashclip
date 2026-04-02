# VLM Prompts

Prompts used in the zero-shot VLM experiments. Two models were evaluated: Gemini 3 Flash and Qwen3-Omni.

## Task A: Aesthetic Score Prediction

### Z1 / Z2 (Zero-shot, video only / video+audio)

The same prompt is used for Z1 (silent video) and Z2 (video with audio). The difference is whether the input video contains an audio track.

```
You are an expert viewer of competitive Super Smash Bros. Ultimate.
Watch this KO clip and rate its aesthetic quality on a scale of 1 to 5.
1 = mundane/routine KO, 5 = spectacular/creative/exciting KO.
Respond with only a single integer from 1 to 5.
```

### Z3 (With metadata context)

A metadata context line is prepended to the Z1/Z2 prompt.

```
Context: {killer} KO'd {victim} with {move} on {stage}.
You are an expert viewer of competitive Super Smash Bros. Ultimate.
Watch this KO clip and rate its aesthetic quality on a scale of 1 to 5.
1 = mundane/routine KO, 5 = spectacular/creative/exciting KO.
Respond with only a single integer from 1 to 5.
```

## Combined Task A+B+C Prompt (Unified)

In the final experiments, Tasks A, B, and C are combined into a single API call per clip.

```
You are an expert viewer of competitive Super Smash Bros. Ultimate.
Watch this KO clip and answer the following three questions.

Q1 (Aesthetic Score): Rate the aesthetic quality of this KO on a scale of 1 to 5.
1 = mundane/routine, 5 = spectacular/creative/exciting.

Q2 (Metadata): Identify the following from the clip.
Killer character (choose one): [{character_list}]
Victim character (choose one): [{character_list}]
Stage (choose one): [{stage_list}]
Finishing move (choose one): [{move_list}]

Q3 (Scene Tags): Select ALL applicable tags. Only select a tag if the clip clearly matches its definition. Most clips match 1-4 tags.
- Combo KO: A KO achieved through a multi-hit combo sequence (3+ connected hits leading to the finishing blow).
- Zero-to-death: A combo or sequence that takes the opponent from 0% damage all the way to a KO without dropping the advantage.
[... all 20 tags with definitions ...]

Respond with ONLY a JSON object: {"score": N, "killer": "...", "victim": "...", "stage": "...", "move": "...", "tags": ["...", ...]}
```

The `{character_list}`, `{stage_list}`, and `{move_list}` are populated from `vocab.json`. Tag definitions are included inline to reduce ambiguity.

## Task C: Scene Tag Prediction (Standalone)

Used in earlier experiments before the combined prompt was adopted.

```
You are an expert viewer of competitive Super Smash Bros. Ultimate.
Watch this KO clip and select ALL applicable scene tags from the following list:
[Combo KO, Zero-to-death, Edgeguarding, One-turn kill,
KO race, Close-range, Ledge trapping, Landing punish, Neutral,
Read, Whiff punish, Approach, Trade, Trap, Mash, Punish,
Notable commentary, Funny moment, Player cam, Misplay]
Respond with only a JSON array of matching tag names. Example: ["Edgeguarding", "Read"]
If none apply, respond with [].
```

## Task D: Rationale Generation

### R1 (Base -- video + score only)

```
You are an expert viewer of competitive Super Smash Bros. Ultimate.
This KO clip was rated {score}/5 for aesthetic quality.
Explain in 2-4 sentences why this clip deserves that score.
Focus on the technical skill, situational context, creativity, and excitement of the KO.
```

### R2 (With oracle metadata)

Ground-truth metadata from the annotation is prepended:

```
Context: {killer} KO'd {victim} with {move} on {stage}.
You are an expert viewer of competitive Super Smash Bros. Ultimate.
This KO clip was rated {score}/5 for aesthetic quality.
Explain in 2-4 sentences why this clip deserves that score.
Focus on the technical skill, situational context, creativity, and excitement of the KO.
```

### R3 (With predicted metadata)

Same format as R2, but the metadata comes from the Z2 zero-shot predictions instead of ground truth. If no prediction is available or parsing failed, falls back to R1.

## Conditions Summary

| Condition | Input | Metadata | Task |
|-----------|-------|----------|------|
| Z1 | Silent video | None | A, B, C |
| Z2 | Video + audio | None | A, B, C |
| Z3 | Video + audio | Ground truth | A |
| R1 | Video + audio | Score only | D |
| R2 | Video + audio | Ground truth | D |
| R3 | Video + audio | Predicted | D |
