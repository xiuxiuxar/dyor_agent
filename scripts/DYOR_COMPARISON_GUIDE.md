# DYOR Agent Template & LLM Comparison System

## 🎯 Overview

This system allows you to test and compare different Jinja2 prompt templates with multiple LLM providers to evaluate the quality of generated DYOR (Do Your Own Research) reports.

## 📁 Directory Structure

```
scripts/
├── test_prompt_template.py              # Main comparison script
├── .env                                  # API keys (create from .env.example)
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
    │   └── ECO_old_template_meta_llama_llama_3_2_90b.md
    └── new_template/
        ├── RENEW_new_template_openai_gpt_4o.md
        ├── RENEW_new_template_anthropic_claude_3_5_sonnet.md
        └── ECO_new_template_meta_llama_llama_3_2_90b.md
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Copy the environment template
cp .env .env.local

# Edit .env.local and add your OpenRouter API key
# Get your key from: https://openrouter.ai/keys
echo "OPENROUTER_API_KEY=your_actual_key_here" > .env
```

### 2. Basic Usage

```bash
# List available assets and models
python test_prompt_template.py --list-assets
python test_prompt_template.py --list-models

# Process templates only (no LLM calls)
python test_prompt_template.py --templates-only

# Full pipeline: templates + LLM generation
python test_prompt_template.py --full-pipeline

# Compare generated reports
python test_prompt_template.py --compare-reports
```

## 📋 Detailed Workflow

### Phase 1: Template Processing

The script processes each JSON file through both templates:

1. **Old Template**: `packages/xiuxiuxar/skills/dyor_app/prompt_template.jinja`
2. **New Template**: `packages/xiuxiuxar/skills/dyor_app/prompt_template_updated.jinja`

```bash
# Process all assets with both templates
python test_prompt_template.py --templates-only

# Process specific asset only
python test_prompt_template.py --templates-only --asset RENEW
```

### Phase 2: LLM Generation

For each rendered template, the script calls multiple LLM models:

- `openai/gpt-4o`
- `anthropic/claude-3-5-sonnet-20241022`
- `meta-llama/llama-3.2-90b-vision-instruct`
- `google/gemini-pro-1.5`
- `mistralai/mistral-large`

```bash
# Generate reports with all models
python test_prompt_template.py --llm-only

# Use specific models only
python test_prompt_template.py --llm-only --models gpt-4o,claude-3-5-sonnet

# Full pipeline with custom models
python test_prompt_template.py --full-pipeline --models gpt-4o,claude-3-5-sonnet
```

### Phase 3: Comparison & Analysis

```bash
# Compare all generated reports
python test_prompt_template.py --compare-reports

# Compare specific asset
python test_prompt_template.py --compare-reports --asset RENEW

# Show detailed differences
python test_prompt_template.py --compare-reports --show-diff
```

## 🔧 Advanced Usage

### Custom Model Selection

```bash
# Use only OpenAI models
python test_prompt_template.py --full-pipeline --models openai/gpt-4o

# Compare specific template types
python test_prompt_template.py --llm-only --template new --models claude-3-5-sonnet
```

### Filtering and Targeting

```bash
# Process specific assets
python test_prompt_template.py --templates-only --asset ECO

# Work with specific template types
python test_prompt_template.py --llm-only --template old

# Generate reports for one asset with one model
python test_prompt_template.py --full-pipeline --asset RENEW --models gpt-4o
```

## 📊 Analysis Workflow

1. **Generate All Combinations**:
   ```bash
   python test_prompt_template.py --full-pipeline
   ```

2. **Review Template Outputs**:
   - Check `report_data_template_filled/` for template quality
   - Compare old vs new template structures

3. **Evaluate LLM Responses**:
   - Review `report_data_llm_output/` for final reports
   - Compare model performance across templates

4. **Identify Best Combinations**:
   ```bash
   python test_prompt_template.py --compare-reports --show-diff
   ```

## 📝 JSON Data Format

Your JSON files should follow this structure:

```json
{
  "asset_id": 25,
  "trigger_id": 78,
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
    "unlocks_recent": [],
    "unlocks_upcoming": [],
    "recent_news": [],
    "official_updates": [],
    "onchain_highlights": []
  }
}
```

## 🎯 Expected Outcomes

### Template Comparison
- **Old Template**: 6 sections (Overview, Key Recent Changes, Recent News/Events, Unlock Events, Analysis, Conclusion)
- **New Template**: 7 sections (adds Community & Social Chatter)

### Model Performance Analysis
- Response quality and consistency
- Adherence to template instructions
- Handling of missing data
- Markdown formatting quality

### Decision Matrix
Use the generated reports to determine:
1. Which template produces better structured prompts
2. Which models perform best with each template
3. Optimal template + model combinations for production

## 🚨 Troubleshooting

### Common Issues

1. **Missing API Key**:
   ```
   ❌ No OpenRouter API key found
   ```
   Solution: Set `OPENROUTER_API_KEY` in your `.env` file

2. **No JSON Files Found**:
   ```
   ❌ No JSON files found in report_data_jsons
   ```
   Solution: Add JSON files to `report_data_jsons/` directory

3. **Template Rendering Errors**:
   ```
   ❌ Error rendering template: 'variable' is undefined
   ```
   Solution: Check JSON structure matches expected template variables

### Debug Mode

Enable verbose logging:
```bash
# Edit the script and change logging level to DEBUG
# Or add debug flags to see more details
python test_prompt_template.py --templates-only 2>&1 | tee debug.log
```

## 💡 Best Practices

1. **Start Small**: Test with 1-2 assets and 1-2 models first
2. **Iterative Testing**: Use `--templates-only` to validate templates before LLM calls
3. **Cost Management**: LLM calls cost money - use `--models` to limit usage
4. **Comparison**: Always run comparison analysis to identify optimal combinations
5. **Documentation**: Keep track of which combinations work best for your use case

## 🔮 Future Enhancements

- [ ] Add report quality scoring metrics
- [ ] Implement cost tracking for API calls
- [ ] Add batch processing for large datasets
- [ ] Create automated A/B testing workflows
- [ ] Generate comparison reports in markdown/HTML
