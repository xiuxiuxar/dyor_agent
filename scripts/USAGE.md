# Prompt Template Testing - Usage Guide

## Overview
Located in `/scripts` folder, this tool processes JSON files containing report data and generates LLM prompts using your Jinja2 template.

## Directory Structure
```
scripts/
├── test_prompt_template.py          # Main script
├── report_data_jsons/              # Input JSON files
│   ├── sample_renew_report.json    # Sample file 1
│   └── ecotoken_report.json        # Sample file 2
└── report_data_jsons_output/       # Generated prompts
    ├── sample_renew_report_output.txt
    └── ecotoken_report_output.txt
```

## Usage Commands

### 1. List Available JSON Files
```bash
python test_prompt_template.py --list-jsons
```

### 2. Process Single JSON File
```bash
# Process specific file from report_data_jsons/
python test_prompt_template.py --json sample_renew_report.json

# Process file with custom output name
python test_prompt_template.py --json sample_renew_report.json --output my_prompt.txt

# Show output in terminal as well
python test_prompt_template.py --json sample_renew_report.json --show-output
```

### 3. Batch Process All JSON Files
```bash
# Process all files in report_data_jsons/ 
# Outputs to report_data_jsons_output/ with "_output.txt" suffix
python test_prompt_template.py --batch

# Show output in terminal for all files
python test_prompt_template.py --batch --show-output
```

### 4. Use Different Template
```bash
# Use old template for comparison
python test_prompt_template.py --json sample_renew_report.json \
  --template ../packages/xiuxiuxar/skills/dyor_app/prompt_template.jinja
```

## JSON File Format
Your JSON files should contain the full structure with `report_data_json`:

```json
{
  "asset_id": 35,
  "trigger_id": 65,
  "report_data_json": {
    "asset_info": { ... },
    "key_metrics": { ... },
    "social_summary": { ... },
    "topic_summary": { ... },
    ...
  },
  "llm_model_used": "Meta-Llama-3-3-70B-Instruct",
  ...
}
```

The script automatically extracts `report_data_json` and processes it with the template.

## Output
- **Single file**: Creates `<filename>_output.txt` in current directory
- **Batch mode**: Creates `report_data_jsons_output/` folder with all outputs
- Each output file contains the complete LLM prompt ready to send to an AI model

## Next Steps
1. Copy your real JSON files to `report_data_jsons/`
2. Run batch processing to generate all prompts
3. Use the generated prompts with your LLM API calls
