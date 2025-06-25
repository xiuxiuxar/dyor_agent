# DYOR Template & LLM Comparison System

## 🎯 Overview

Production-ready script for testing and comparing Jinja2 prompt templates with multiple LLM providers. Supports side-by-side comparison of old vs new templates across different AI models to evaluate DYOR (Do Your Own Research) report quality.

## ✅ What's Working

✅ **Template Processing** - Both old and new Jinja2 templates work with your JSON data  
✅ **Path Detection** - Works from project root with `poetry run`  
✅ **JSON Data Loading** - Handles nested `report_data_json` structure  
✅ **Simple Test Mode** - Easy one-command testing with cost controls  
✅ **LLM Integration** - Ready for OpenRouter API with multiple models  
✅ **Batch Processing** - Can process multiple assets and compare results  

## 📁 Directory Structure

```
scripts/
├── test_prompt_template.py              # Main comparison script
├── README.md                            # This file
├── report_data_jsons/                   # Source JSON files
│   ├── sample_renew_report.json         # Sample ReNeW data
│   └── ecotoken_report.json             # Sample EcoToken data
├── report_data_template_filled/         # Rendered Jinja templates
│   ├── old_template/
│   │   ├── RENEW_old_template.txt       # Old template + RENEW data
│   │   └── ECO_old_template.txt         # Old template + ECO data
│   └── new_template/
│       ├── RENEW_new_template.txt       # New template + RENEW data
│       └── ECO_new_template.txt         # New template + ECO data
└── report_data_llm_output/              # Final LLM-generated reports
    ├── old_template/
    │   ├── RENEW_old_template_openai_gpt_4o.md
    │   ├── RENEW_old_template_anthropic_claude_3_5_sonnet.md
    │   └── ECO_old_template_*.md
    └── new_template/
        ├── RENEW_new_template_openai_gpt_4o.md
        ├── RENEW_new_template_anthropic_claude_3_5_sonnet.md
        └── ECO_new_template_*.md
```

## 🚀 Quick Start

### 1. Test Templates Only (No Cost)

```bash
# Test that templates work with your JSON data (no API calls)
poetry run python scripts/test_prompt_template.py --templates-only

# Check the generated template-filled prompts
ls scripts/report_data_template_filled/*/
```

### 2. Full Test with AI Models

```bash
# Set your OpenRouter API key (get from https://openrouter.ai/keys)
export OPENROUTER_API_KEY="sk-or-your_actual_key_here"

# Run simple test (uses 2 models, costs ~$2-4)
poetry run python scripts/test_prompt_template.py --simple-test
```

## 📋 Usage Commands

### Basic Operations

```bash
# List available assets and models
poetry run python scripts/test_prompt_template.py --list-assets
poetry run python scripts/test_prompt_template.py --list-models

# Process templates only (no LLM calls)
poetry run python scripts/test_prompt_template.py --templates-only

# Full pipeline: templates + LLM generation (uses all 5 models)
poetry run python scripts/test_prompt_template.py --full-pipeline

# Simple test: one asset with 2 models (cost-controlled)
poetry run python scripts/test_prompt_template.py --simple-test
```

### Targeted Processing

```bash
# Process specific asset only
poetry run python scripts/test_prompt_template.py --templates-only --asset ECO

# Use specific models only
poetry run python scripts/test_prompt_template.py --llm-only --models gpt-4o,claude-3-5-sonnet

# Process specific template type
poetry run python scripts/test_prompt_template.py --llm-only --template new

# Combined: specific asset + models
poetry run python scripts/test_prompt_template.py --simple-test --asset RENEW --models gpt-4o
```

### Analysis & Comparison

```bash
# Compare all generated reports
poetry run python scripts/test_prompt_template.py --compare-reports

# Compare specific asset with detailed differences
poetry run python scripts/test_prompt_template.py --compare-reports --asset ECO --show-diff

# Get help with all options
poetry run python scripts/test_prompt_template.py --help
```

## 🤖 Available Models

The script supports these LLM models via OpenRouter:

- `openai/gpt-4o` - OpenAI's latest GPT-4
- `anthropic/claude-3-5-sonnet-20241022` - Anthropic's Claude 3.5

(More models can be added by modifying the `CONFIG['models']` list in the script)

## 📊 Workflow

### Phase 1: Template Processing
1. Script loads JSON files from `report_data_jsons/`
2. Extracts `report_data_json` from each file
3. Renders both old and new Jinja2 templates with the data
4. Saves rendered prompts to `report_data_template_filled/`

### Phase 2: LLM Generation
1. Reads rendered templates
2. Sends each to multiple LLM models via OpenRouter API
3. Saves AI-generated reports to `report_data_llm_output/`

### Phase 3: Comparison
1. Analyzes generated reports
2. Shows size differences between template outputs
3. Provides side-by-side comparison data

## 📝 JSON Data Format

Your JSON files should follow this structure:

```json
{
  "asset_id": 35,
  "trigger_id": 65,
  "report_data_json": {
    "asset_info": {
      "name": "TokenName",
      "symbol": "TKN",
      "market_cap": 1000000,
      "market_cap_rank": 1000,
      "category": "DeFi",
      "contract_address": "0x..."
    },
    "key_metrics": {
      "price_change_24h": 5.2,
      "volume_change_24h": 15.3,
      "mindshare": 2.5,
      "mindshare_24h": 1.2
    },
    "social_summary": {
      "sentiment_score": 88.0,
      "mention_change_24h": 75.0
    },
    "topic_summary": "Token description and social chatter overview",
    "unlocks_recent": [],
    "unlocks_upcoming": [],
    "recent_news": [],
    "official_updates": [],
    "onchain_highlights": []
  },
  "llm_model_used": "Meta-Llama-3-3-70B-Instruct"
}
```

The script automatically extracts `report_data_json` and processes it.

## 🔧 Configuration

### Environment Setup

The script uses `load_dotenv()` to automatically find your `.env` file. Create one in the project root:

```bash
# In project root (not scripts folder)
echo "OPENROUTER_API_KEY=your_actual_key_here" > .env
```

### Template Paths

Templates are automatically located at:
- **Old**: `packages/xiuxiuxar/skills/dyor_app/prompt_template.jinja`
- **New**: `packages/xiuxiuxar/skills/dyor_app/prompt_template_updated.jinja`

## 🎯 Analysis Workflow

1. **Quick Validation**:
   ```bash
   poetry run python scripts/test_prompt_template.py --templates-only
   ```

2. **Cost-Controlled Testing**:
   ```bash
   poetry run python scripts/test_prompt_template.py --simple-test
   ```

3. **Review Results**:
   - Check `report_data_template_filled/` for template quality
   - Review `report_data_llm_output/` for final AI reports

4. **Compare & Choose**:
   ```bash
   poetry run python scripts/test_prompt_template.py --compare-reports --show-diff
   ```

## 📈 Expected Outcomes

### Template Comparison
- **Old Template**: 6 sections (Overview, Key Recent Changes, Recent News/Events, Unlock Events, Analysis, Conclusion)
- **New Template**: 7 sections (adds Community & Social Chatter)

### Model Performance
Different models will produce varying quality reports. The comparison system helps identify the best template + model combination for your use case.

## 🚨 Cost Management

- **Templates Only**: Free (no API calls)
- **Simple Test**: ~$2-4 (2 models × 2 templates × assets)
- **Full Pipeline**: ~$10-20 (5 models × 2 templates × assets)

Use `--simple-test` for cost-controlled evaluation before running full comparisons.

## 🎯 Next Steps

1. Run `--templates-only` to verify templates work with your data
2. Add your OpenRouter API key to `.env` file
3. Run `--simple-test` to see AI-generated reports
4. Compare results to choose the best template + model combination
5. Use the winning combination in your production system! 