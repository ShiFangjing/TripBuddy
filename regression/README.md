# Golden Prompt Regression (Phase-1)

This folder stores phase-1 golden prompts for new/legacy comparison.

## Files
- `golden_prompts.json`: 10 agreed prompts and their test focus tags.

## How to use
1. Run legacy CLI on each prompt and record key fields:
   - `agent_schedule`
   - `preferences` write-back
   - `trip extraction`
2. Run `cli_langgraph.py --query "..."` on each prompt.
3. Compare behavioral parity and explain any differences.

## Notes
- Phase-1 only migrates `event_collection` + `preference` execution.
- Other agents are explicit placeholders and expected to return `not_implemented` when scheduled.

## Automated runner
```bash
python regression/run_phase1_regression.py
```

Output:
- `regression/phase1_report.json`
