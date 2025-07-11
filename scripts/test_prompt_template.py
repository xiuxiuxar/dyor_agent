#!/usr/bin/env python3
"""
DYOR Agent Template & LLM Comparison System
===========================================

Production-ready script for testing and comparing prompt templates with multiple LLM providers.
Supports side-by-side comparison of old vs new Jinja templates across different AI models.

DIRECTORY STRUCTURE:
-------------------
scripts/
├── test_prompt_template.py              # This script
├── report_data_jsons/                   # Source JSON files
│   ├── sample_renew_report.json
│   └── ecotoken_report.json
├── report_data_template_filled/         # Rendered Jinja templates
│   ├── old_template/
│   │   ├── RENEW_old_template.txt
│   │   └── ECO_old_template.txt
│   └── new_template/
│       ├── RENEW_new_template.txt
│       └── ECO_new_template.txt
└── report_data_llm_output/              # Final LLM-generated reports
    ├── old_template/
    │   ├── RENEW_old_template_gpt4.md
    │   ├── RENEW_old_template_claude.md
    │   └── RENEW_old_template_llama.md
    └── new_template/
        ├── RENEW_new_template_gpt4.md
        ├── RENEW_new_template_claude.md
        └── RENEW_new_template_llama.md

USAGE EXAMPLES:
--------------

1. Full Pipeline (Templates + LLM Generation):
   python test_prompt_template.py --full-pipeline
   python test_prompt_template.py --full-pipeline --models gpt-4,claude-3-sonnet

2. Template Processing Only:
   python test_prompt_template.py --templates-only
   python test_prompt_template.py --templates-only --asset RENEW

3. LLM Generation Only (assumes templates exist):
   python test_prompt_template.py --llm-only --models gpt-4
   python test_prompt_template.py --llm-only --template new --models claude-3-sonnet

4. Compare Existing Reports:
   python test_prompt_template.py --compare-reports --asset RENEW
   python test_prompt_template.py --compare-reports --show-diff

5. List Available Assets/Models:
   python test_prompt_template.py --list-assets
   python test_prompt_template.py --list-models

CONFIGURATION:
-------------
Set your OpenRouter API key:
export OPENROUTER_API_KEY="your_key_here"

Or create a .env file in the scripts directory:
OPENROUTER_API_KEY=your_key_here

WORKFLOW:
--------
1. Add JSON files to report_data_jsons/
2. Run: python test_prompt_template.py --full-pipeline
3. Compare outputs in report_data_llm_output/
4. Analyze template performance differences
"""

import json
import sys
import argparse
import asyncio
import aiohttp
import os
from datetime import datetime
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from typing import Dict, List, Optional, Tuple
import time
from dataclasses import dataclass
import logging
from dotenv import load_dotenv

# Configuration
def get_config():
    """Get configuration with correct paths"""
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent  # Go up from scripts/ to project root
    
    return {
        'source_jsons': script_dir / 'report_data_jsons',
        'template_filled': script_dir / 'report_data_template_filled', 
        'llm_output': script_dir / 'report_data_llm_output',
        'old_template': project_root / 'packages' / 'xiuxiuxar' / 'skills' / 'dyor_app' / 'prompt_template.jinja',
        'new_template': project_root / 'packages' / 'xiuxiuxar' / 'skills' / 'dyor_app' / 'prompt_template_updated.jinja'
    }

CONFIG = {
    'paths': get_config(),
    'models': [
        'openai/gpt-4o',
        'anthropic/claude-3-5-sonnet-20241022',
        # Free/Low-cost models
        'google/gemini-2.5-flash-lite-preview-06-17',
        'meta-llama/llama-4-maverick-17b-128e-instruct',
        'meta-llama/llama-3.2-90b-vision-instruct',
        'meta-llama/llama-3.1-70b-instruct', 
        'deepseek/deepseek-chat',
        'google/gemma-2-9b-it',
        'microsoft/wizardlm-2-8x22b',
        'qwen/qwen-2.5-72b-instruct'
    ],
    'evaluator_model': 'openai/gpt-4o',  # Model used for template evaluation
    'api': {
        'openrouter_base_url': 'https://openrouter.ai/api/v1',
        'max_retries': 3,
        'retry_delay': 2,
        'timeout': 120
    }
}

@dataclass
class ProcessingResult:
    success: bool
    asset_symbol: str
    template_type: str
    model_name: str = None
    output_path: Path = None
    error: str = None
    processing_time: float = 0

class DYORComparison:
    def __init__(self):
        # Load .env file from project root
        load_dotenv()
        self.setup_logging()
        self.api_key = self.get_api_key()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('dyor_comparison.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_api_key(self) -> Optional[str]:
        """Get OpenRouter API key from environment variables"""
        api_key = os.getenv('OPENROUTER_API_KEY')
        if api_key and api_key not in ['placeholder', 'your_key_here', 'sk-or-your_key_here', 'your_actual_key_here']:
            return api_key
        return None

    def timestamp_to_date(self, timestamp_str):
        """Convert timestamp string to readable date format"""
        try:
            if isinstance(timestamp_str, (int, float)):
                dt = datetime.fromtimestamp(timestamp_str)
            else:
                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return dt.strftime("%B %d, %Y")
        except:
            return str(timestamp_str)

    def intcomma(self, value):
        """Add commas to numbers for readability"""
        try:
            return f"{float(value):,.0f}"
        except:
            return str(value)

    def extract_asset_symbol(self, data: Dict) -> str:
        """Extract asset symbol from JSON data"""
        try:
            if "report_data_json" in data:
                asset_info = data["report_data_json"].get("asset_info", {})
            else:
                asset_info = data.get("asset_info", {})
            
            symbol = asset_info.get("symbol", "UNKNOWN")
            return symbol.upper()
        except Exception as e:
            self.logger.warning(f"Could not extract asset symbol: {e}")
            return "UNKNOWN"

    def load_json_data(self, json_path: Path) -> Tuple[Dict, str]:
        """Load JSON data and extract asset symbol"""
        try:
            with open(json_path, 'r') as f:
                full_data = json.load(f)
            
            # Extract report data
            if "report_data_json" in full_data:
                data = full_data["report_data_json"]
                self.logger.info(f"✓ Extracted report_data_json from {json_path.name}")
            else:
                data = full_data
                self.logger.info(f"✓ Using {json_path.name} as direct report data")
            
            # Convert topic_summary dict to string if needed
            if "topic_summary" in data and isinstance(data["topic_summary"], dict):
                data["topic_summary"] = data["topic_summary"].get("overview", "")
            
            asset_symbol = self.extract_asset_symbol({"report_data_json": data})
            return data, asset_symbol
            
        except Exception as e:
            self.logger.error(f"❌ Error loading {json_path.name}: {e}")
            return None, None

    def render_template(self, template_path: Path, data: Dict) -> Optional[str]:
        """Render a Jinja2 template with given data"""
        if not template_path.exists():
            self.logger.error(f"❌ Template not found: {template_path}")
            return None

        try:
            # Setup Jinja2 environment
            env = Environment(loader=FileSystemLoader(template_path.parent))
            env.filters['timestamp_to_date'] = self.timestamp_to_date
            env.filters['intcomma'] = self.intcomma
            
            template = env.get_template(template_path.name)
            rendered = template.render(**data)
            return rendered
            
        except Exception as e:
            self.logger.error(f"❌ Error rendering template {template_path.name}: {e}")
            return None

    def process_templates(self, asset_filter: Optional[str] = None) -> List[ProcessingResult]:
        """Process all JSON files through both templates"""
        self.logger.info("🚀 Starting template processing...")
        
        source_dir = CONFIG['paths']['source_jsons']
        if not source_dir.exists():
            self.logger.error(f"❌ Source directory not found: {source_dir}")
            return []

        json_files = list(source_dir.glob("*.json"))
        if not json_files:
            self.logger.error(f"❌ No JSON files found in {source_dir}")
            return []

        results = []
        
        for json_file in json_files:
            data, asset_symbol = self.load_json_data(json_file)
            if not data or not asset_symbol:
                continue
                
            # Filter by asset if specified
            if asset_filter and asset_symbol != asset_filter.upper():
                continue
                
            self.logger.info(f"📄 Processing {asset_symbol} from {json_file.name}")
            
            # Process both templates
            for template_type, template_path in [
                ('old', CONFIG['paths']['old_template']),
                ('new', CONFIG['paths']['new_template'])
            ]:
                start_time = time.time()
                
                rendered = self.render_template(template_path, data)
                if rendered:
                    # Save rendered template
                    output_dir = CONFIG['paths']['template_filled'] / f"{template_type}_template"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    output_file = output_dir / f"{asset_symbol}_{template_type}_template.txt"
                    with open(output_file, 'w') as f:
                        f.write(rendered)
                    
                    processing_time = time.time() - start_time
                    result = ProcessingResult(
                        success=True,
                        asset_symbol=asset_symbol,
                        template_type=template_type,
                        output_path=output_file,
                        processing_time=processing_time
                    )
                    results.append(result)
                    self.logger.info(f"   ✓ {template_type.title()} template → {output_file}")
                else:
                    result = ProcessingResult(
                        success=False,
                        asset_symbol=asset_symbol,
                        template_type=template_type,
                        error=f"Failed to render {template_type} template"
                    )
                    results.append(result)

        return results

    async def call_llm_api(self, session: aiohttp.ClientSession, model: str, prompt: str) -> Optional[str]:
        """Make API call to OpenRouter"""
        if not self.api_key:
            self.logger.error("❌ No OpenRouter API key found")
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/trend-spotter/dyor-agent",
            "X-Title": "DYOR Agent Template Comparison"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4000,
            "temperature": 0.7
        }

        for attempt in range(CONFIG['api']['max_retries']):
            try:
                async with session.post(
                    f"{CONFIG['api']['openrouter_base_url']}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=CONFIG['api']['timeout'])
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data['choices'][0]['message']['content']
                    else:
                        error_text = await response.text()
                        self.logger.error(f"❌ API error {response.status}: {error_text}")
                        
            except Exception as e:
                self.logger.error(f"❌ API call attempt {attempt + 1} failed: {e}")
                if attempt < CONFIG['api']['max_retries'] - 1:
                    await asyncio.sleep(CONFIG['api']['retry_delay'] * (attempt + 1))

        return None

    async def process_llm_generation(self, models: List[str], template_filter: Optional[str] = None, asset_filter: Optional[str] = None) -> List[ProcessingResult]:
        """Generate LLM reports from template-filled prompts"""
        self.logger.info("🤖 Starting LLM generation...")
        
        template_dir = CONFIG['paths']['template_filled']
        if not template_dir.exists():
            self.logger.error(f"❌ Template-filled directory not found: {template_dir}")
            return []

        results = []
        
        # Find all template files
        template_types = ['old', 'new'] if not template_filter else [template_filter]
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for template_type in template_types:
                template_subdir = template_dir / f"{template_type}_template"
                if not template_subdir.exists():
                    continue
                    
                for template_file in template_subdir.glob("*.txt"):
                    # Extract asset symbol from filename
                    asset_symbol = template_file.stem.replace(f"_{template_type}_template", "")
                    
                    # Filter by asset if specified
                    if asset_filter and asset_symbol != asset_filter.upper():
                        continue
                    
                    # Read the rendered template
                    with open(template_file, 'r') as f:
                        prompt = f.read()
                    
                    # Create tasks for each model
                    for model in models:
                        task = self.generate_single_report(
                            session, asset_symbol, template_type, model, prompt
                        )
                        tasks.append(task)
            
            # Execute all tasks concurrently
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                # Filter out exceptions and convert to ProcessingResult objects
                results = [r for r in results if isinstance(r, ProcessingResult)]

        return results

    async def generate_single_report(self, session: aiohttp.ClientSession, asset_symbol: str, template_type: str, model: str, prompt: str) -> ProcessingResult:
        """Generate a single LLM report"""
        start_time = time.time()
        
        try:
            self.logger.info(f"🔄 Generating {asset_symbol} / {template_type} / {model}")
            
            response = await self.call_llm_api(session, model, prompt)
            
            if response:
                # Save the response
                output_dir = CONFIG['paths']['llm_output'] / f"{template_type}_template"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                model_safe = model.replace('/', '_').replace('-', '_')
                output_file = output_dir / f"{asset_symbol}_{template_type}_template_{model_safe}.md"
                
                with open(output_file, 'w') as f:
                    f.write(response)
                
                processing_time = time.time() - start_time
                self.logger.info(f"   ✓ Generated → {output_file}")
                
                return ProcessingResult(
                    success=True,
                    asset_symbol=asset_symbol,
                    template_type=template_type,
                    model_name=model,
                    output_path=output_file,
                    processing_time=processing_time
                )
            else:
                return ProcessingResult(
                    success=False,
                    asset_symbol=asset_symbol,
                    template_type=template_type,
                    model_name=model,
                    error="Failed to get LLM response"
                )
                
        except Exception as e:
            return ProcessingResult(
                success=False,
                asset_symbol=asset_symbol,
                template_type=template_type,
                model_name=model,
                error=str(e)
            )

    def list_available_assets(self) -> List[str]:
        """List available assets from JSON files"""
        source_dir = CONFIG['paths']['source_jsons']
        if not source_dir.exists():
            return []
        
        assets = []
        for json_file in source_dir.glob("*.json"):
            _, asset_symbol = self.load_json_data(json_file)
            if asset_symbol:
                assets.append(asset_symbol)
        
        return sorted(set(assets))

    async def evaluate_template_comparison(self, old_report_path: Path, new_report_path: Path, asset: str, model: str) -> Dict:
        """Compare old vs new template for the same asset+model using GPT-4o"""
        try:
            with open(old_report_path, 'r') as f:
                old_content = f.read()
            with open(new_report_path, 'r') as f:
                new_content = f.read()
            
            evaluation_prompt = f"""
You are an expert financial analyst evaluating DYOR (Do Your Own Research) crypto reports. 

Compare these two reports for {asset} generated by {model}. Both analyze the same underlying data but use different prompt templates.

OLD TEMPLATE REPORT:
====================
{old_content}

NEW TEMPLATE REPORT:
====================
{new_content}

EVALUATION CRITERIA:
1. **Market Catalyst Focus**: How well does it identify price-moving events and catalysts?
2. **Actionability**: Does it provide clear, specific investment implications vs generic analysis?
3. **Structure & Flow**: Is the information well-organized and easy to follow?
4. **Data Integration**: How effectively does it use and synthesize the available data?
5. **Unique Insights**: Does it provide novel analysis rather than repetitive content?

Please provide:
1. **Winner**: Which template produces a better report? (OLD/NEW/TIE)
2. **Confidence**: How confident are you? (High/Medium/Low)
3. **Scores**: Rate each template 1-10 on each criterion
4. **Key Differences**: What are the 3 most important differences?
5. **Reasoning**: Detailed explanation of your decision

Format your response as JSON:
{{
    "winner": "OLD/NEW/TIE",
    "confidence": "High/Medium/Low", 
    "old_scores": {{
        "market_catalyst": 8,
        "actionability": 7,
        "structure": 9,
        "data_integration": 8,
        "unique_insights": 6
    }},
    "new_scores": {{
        "market_catalyst": 9,
        "actionability": 8,
        "structure": 8,
        "data_integration": 9,
        "unique_insights": 8
    }},
    "key_differences": [
        "New template provides clearer market catalyst timeline",
        "Old template has better risk assessment section",
        "New template integrates social sentiment more effectively"
    ],
    "reasoning": "The new template wins because..."
}}
"""
            
            async with aiohttp.ClientSession() as session:
                result = await self.call_llm_api(session, CONFIG['evaluator_model'], evaluation_prompt)
                if result:
                    try:
                        # Extract JSON from response
                        import re
                        json_match = re.search(r'\{.*\}', result, re.DOTALL)
                        if json_match:
                            evaluation = json.loads(json_match.group())
                            evaluation['asset'] = asset
                            evaluation['model'] = model
                            evaluation['old_report'] = old_report_path.name
                            evaluation['new_report'] = new_report_path.name
                            return evaluation
                    except json.JSONDecodeError:
                        self.logger.error(f"Failed to parse evaluation JSON for {asset} {model}")
                        
        except Exception as e:
            self.logger.error(f"Error evaluating {asset} {model}: {e}")
            
        return None

    async def evaluate_report_quality(self, report_path: Path, asset: str, model: str, template: str) -> Dict:
        """Evaluate individual report quality on a 1-10 scale"""
        try:
            with open(report_path, 'r') as f:
                content = f.read()
            
            quality_prompt = f"""
You are an expert financial analyst evaluating a DYOR (Do Your Own Research) crypto report quality.

REPORT TO EVALUATE:
==================
Asset: {asset}
Model: {model}  
Template: {template}

{content}

QUALITY SCORING (1-10 scale):
1. **Clarity & Writing** (0-2 points): Professional, clear communication
2. **Market Analysis** (0-3 points): Depth and accuracy of price/trend analysis  
3. **Risk Assessment** (0-2 points): Balanced view of risks and opportunities
4. **Actionability** (0-2 points): Clear, specific investment implications
5. **Data Usage** (0-1 point): Effective use of provided data

EVALUATION GUIDELINES:
- Score each category based on how well the report performs
- Consider the report's usefulness for actual investment decisions
- Evaluate factual accuracy and logical reasoning
- Look for specific, actionable insights vs generic statements

Format your response as JSON:
{{
    "scores": {{
        "clarity_writing": 8,
        "market_analysis": 7,
        "risk_assessment": 8,
        "actionability": 6,
        "data_usage": 9
    }},
    "total_score": 38,
    "normalized_score": 7.6,
    "strengths": [
        "Excellent data integration",
        "Clear writing style"
    ],
    "weaknesses": [
        "Limited actionable insights",
        "Generic risk assessment"
    ],
    "overall_assessment": "Detailed reasoning for the score..."
}}
"""
            
            async with aiohttp.ClientSession() as session:
                result = await self.call_llm_api(session, CONFIG['evaluator_model'], quality_prompt)
                if result:
                    try:
                        import re
                        json_match = re.search(r'\{.*\}', result, re.DOTALL)
                        if json_match:
                            evaluation = json.loads(json_match.group())
                            evaluation['asset'] = asset
                            evaluation['model'] = model
                            evaluation['template'] = template
                            evaluation['report_file'] = report_path.name
                            return evaluation
                    except json.JSONDecodeError:
                        self.logger.error(f"Failed to parse quality JSON for {report_path.name}")
                        
        except Exception as e:
            self.logger.error(f"Error evaluating quality for {report_path.name}: {e}")
            
        return None

    async def run_template_evaluation(self, asset_filter: Optional[str] = None) -> Dict:
        """Run comprehensive template evaluation with quality scoring"""
        llm_dir = CONFIG['paths']['llm_output']
        evaluations_dir = CONFIG['paths']['llm_output'] / 'evaluations'
        evaluations_dir.mkdir(exist_ok=True)
        
        if not llm_dir.exists():
            self.logger.error("❌ No LLM output directory found. Run --full-pipeline first.")
            return {}

        # Find all reports
        old_reports = list((llm_dir / "old_template").glob("*.md"))
        new_reports = list((llm_dir / "new_template").glob("*.md"))
        
        if asset_filter:
            old_reports = [r for r in old_reports if asset_filter.upper() in r.name]
            new_reports = [r for r in new_reports if asset_filter.upper() in r.name]

        print(f"🤖 Starting Template Evaluation with {CONFIG['evaluator_model']}")
        print("=" * 60)
        print(f"Old template reports: {len(old_reports)}")
        print(f"New template reports: {len(new_reports)}")
        
        # Group reports for comparison
        comparisons = {}
        all_reports = []
        
        # Collect old reports
        for report in old_reports:
            parts = report.stem.split('_')
            if len(parts) >= 4:
                asset_name = parts[0]
                model_name = '_'.join(parts[3:])
                key = f"{asset_name}_{model_name}"
                if key not in comparisons:
                    comparisons[key] = {}
                comparisons[key]['old'] = report
                all_reports.append({'path': report, 'asset': asset_name, 'model': model_name, 'template': 'old'})

        # Collect new reports  
        for report in new_reports:
            parts = report.stem.split('_')
            if len(parts) >= 4:
                asset_name = parts[0]
                model_name = '_'.join(parts[3:])
                key = f"{asset_name}_{model_name}"
                if key not in comparisons:
                    comparisons[key] = {}
                comparisons[key]['new'] = report
                all_reports.append({'path': report, 'asset': asset_name, 'model': model_name, 'template': 'new'})

        # Run evaluations
        template_comparisons = []
        quality_evaluations = []
        
        print(f"\n🔍 Running {len([k for k, v in comparisons.items() if 'old' in v and 'new' in v])} template comparisons...")
        
        # Template comparisons (old vs new)
        for key, reports in comparisons.items():
            if 'old' in reports and 'new' in reports:
                asset_name = key.split('_')[0]
                model_name = '_'.join(key.split('_')[1:])
                print(f"   Comparing {asset_name} with {model_name}...")
                
                comparison = await self.evaluate_template_comparison(
                    reports['old'], reports['new'], asset_name, model_name
                )
                if comparison:
                    template_comparisons.append(comparison)

        print(f"\n📊 Running {len(all_reports)} quality evaluations...")
        
        # Individual quality evaluations
        for report_info in all_reports:
            print(f"   Evaluating {report_info['asset']} {report_info['template']} {report_info['model']}...")
            
            quality = await self.evaluate_report_quality(
                report_info['path'], report_info['asset'], 
                report_info['model'], report_info['template']
            )
            if quality:
                quality_evaluations.append(quality)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save template comparisons
        comparisons_file = evaluations_dir / f'template_comparisons_{timestamp}.json'
        with open(comparisons_file, 'w') as f:
            json.dump(template_comparisons, f, indent=2)
        
        # Save quality evaluations  
        quality_file = evaluations_dir / f'quality_evaluations_{timestamp}.json'
        with open(quality_file, 'w') as f:
            json.dump(quality_evaluations, f, indent=2)
        
        # Generate summary reports
        self.generate_evaluation_summary(template_comparisons, quality_evaluations, evaluations_dir, timestamp)
        
        return {
            'template_comparisons': template_comparisons,
            'quality_evaluations': quality_evaluations,
            'files': {
                'comparisons': comparisons_file,
                'quality': quality_file
            }
        }

    def generate_evaluation_summary(self, template_comparisons: List[Dict], quality_evaluations: List[Dict], 
                                   output_dir: Path, timestamp: str):
        """Generate comprehensive evaluation summary reports"""
        
        # Template Comparison Summary
        summary_md = output_dir / f'EVALUATION_SUMMARY_{timestamp}.md'
        
        with open(summary_md, 'w') as f:
            f.write(f"# DYOR Template Evaluation Summary\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Evaluator Model: {CONFIG['evaluator_model']}\n\n")
            
            # Template winner analysis
            if template_comparisons:
                f.write("## 🏆 Template Comparison Results\n\n")
                
                winners = {'OLD': 0, 'NEW': 0, 'TIE': 0}
                high_confidence = 0
                
                for comp in template_comparisons:
                    winners[comp['winner']] += 1
                    if comp['confidence'] == 'High':
                        high_confidence += 1
                
                total = len(template_comparisons)
                f.write(f"**Total Comparisons:** {total}\n")
                f.write(f"**High Confidence Evaluations:** {high_confidence}/{total} ({high_confidence/total*100:.1f}%)\n\n")
                
                f.write("### Winner Breakdown:\n")
                for winner, count in winners.items():
                    percentage = count/total*100 if total > 0 else 0
                    f.write(f"- **{winner} Template:** {count}/{total} ({percentage:.1f}%)\n")
                
                f.write("\n### Detailed Comparisons:\n\n")
                for comp in template_comparisons:
                    f.write(f"#### {comp['asset']} - {comp['model']}\n")
                    f.write(f"**Winner:** {comp['winner']} (Confidence: {comp['confidence']})\n\n")
                    
                    f.write("**Scores:**\n")
                    f.write("| Criteria | Old | New |\n")
                    f.write("|----------|-----|-----|\n")
                    for criteria in ['market_catalyst', 'actionability', 'structure', 'data_integration', 'unique_insights']:
                        old_score = comp['old_scores'].get(criteria, 'N/A')
                        new_score = comp['new_scores'].get(criteria, 'N/A')
                        f.write(f"| {criteria.replace('_', ' ').title()} | {old_score} | {new_score} |\n")
                    
                    f.write(f"\n**Key Differences:**\n")
                    for diff in comp.get('key_differences', []):
                        f.write(f"- {diff}\n")
                    
                    f.write(f"\n**Reasoning:** {comp.get('reasoning', 'N/A')}\n\n")
            
            # Quality rankings
            if quality_evaluations:
                f.write("## 📊 Quality Rankings\n\n")
                
                # Sort by normalized score
                sorted_evals = sorted(quality_evaluations, key=lambda x: x.get('normalized_score', 0), reverse=True)
                
                f.write("### Top 10 Highest Quality Reports:\n\n")
                f.write("| Rank | Asset | Template | Model | Score | Strengths |\n")
                f.write("|------|-------|----------|-------|-------|----------|\n")
                
                for i, eval_data in enumerate(sorted_evals[:10], 1):
                    asset = eval_data.get('asset', 'N/A')
                    template = eval_data.get('template', 'N/A')
                    model = eval_data.get('model', 'N/A')
                    score = eval_data.get('normalized_score', 0)
                    strengths = ', '.join(eval_data.get('strengths', [])[:2])
                    f.write(f"| {i} | {asset} | {template} | {model} | {score:.1f} | {strengths} |\n")
                
                # Template performance comparison
                f.write("\n### Template Performance Analysis:\n\n")
                old_scores = [e['normalized_score'] for e in quality_evaluations if e.get('template') == 'old']
                new_scores = [e['normalized_score'] for e in quality_evaluations if e.get('template') == 'new']
                
                if old_scores and new_scores:
                    old_avg = sum(old_scores) / len(old_scores)
                    new_avg = sum(new_scores) / len(new_scores)
                    
                    f.write(f"- **Old Template Average Score:** {old_avg:.2f} ({len(old_scores)} reports)\n")
                    f.write(f"- **New Template Average Score:** {new_avg:.2f} ({len(new_scores)} reports)\n")
                    f.write(f"- **Improvement:** {new_avg - old_avg:+.2f} points\n\n")
                
                # Model performance
                f.write("### Model Performance Analysis:\n\n")
                model_scores = {}
                for eval_data in quality_evaluations:
                    model = eval_data.get('model', 'Unknown')
                    score = eval_data.get('normalized_score', 0)
                    if model not in model_scores:
                        model_scores[model] = []
                    model_scores[model].append(score)
                
                model_averages = {model: sum(scores)/len(scores) for model, scores in model_scores.items()}
                sorted_models = sorted(model_averages.items(), key=lambda x: x[1], reverse=True)
                
                f.write("| Rank | Model | Avg Score | Reports |\n")
                f.write("|------|-------|-----------|----------|\n")
                for i, (model, avg_score) in enumerate(sorted_models, 1):
                    count = len(model_scores[model])
                    f.write(f"| {i} | {model} | {avg_score:.2f} | {count} |\n")
        
        print(f"\n✅ Evaluation complete!")
        print(f"📄 Summary report: {summary_md}")
        print(f"📁 All evaluation files: {output_dir}")

    def compare_reports(self, asset: Optional[str] = None, show_diff: bool = False):
        """Compare generated reports between templates"""
        self.logger.info("📊 Comparing reports...")
        
        llm_dir = CONFIG['paths']['llm_output']
        if not llm_dir.exists():
            self.logger.error("❌ No LLM output directory found. Run --full-pipeline first.")
            return

        # Find all reports
        old_reports = list((llm_dir / "old_template").glob("*.md"))
        new_reports = list((llm_dir / "new_template").glob("*.md"))
        
        if asset:
            old_reports = [r for r in old_reports if asset.upper() in r.name]
            new_reports = [r for r in new_reports if asset.upper() in r.name]

        print(f"\n📊 Report Comparison Summary")
        print("=" * 50)
        print(f"Old template reports: {len(old_reports)}")
        print(f"New template reports: {len(new_reports)}")
        
        # Group by asset and model for comparison
        comparisons = {}
        
        for report in old_reports:
            parts = report.stem.split('_')
            if len(parts) >= 4:
                asset_name = parts[0]
                model_name = '_'.join(parts[3:])
                key = f"{asset_name}_{model_name}"
                if key not in comparisons:
                    comparisons[key] = {}
                comparisons[key]['old'] = report

        for report in new_reports:
            parts = report.stem.split('_')
            if len(parts) >= 4:
                asset_name = parts[0]
                model_name = '_'.join(parts[3:])
                key = f"{asset_name}_{model_name}"
                if key not in comparisons:
                    comparisons[key] = {}
                comparisons[key]['new'] = report

        # Display comparisons
        for key, reports in comparisons.items():
            if 'old' in reports and 'new' in reports:
                print(f"\n🔍 {key}:")
                print(f"   Old: {reports['old'].name}")
                print(f"   New: {reports['new'].name}")
                
                if show_diff:
                    # Show basic stats
                    old_size = reports['old'].stat().st_size
                    new_size = reports['new'].stat().st_size
                    print(f"   Size difference: {new_size - old_size:+d} bytes")

    def print_summary(self, results: List[ProcessingResult]):
        """Print processing summary"""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print(f"\n✅ Processing Summary")
        print("=" * 30)
        print(f"Total operations: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            avg_time = sum(r.processing_time for r in successful) / len(successful)
            print(f"Average processing time: {avg_time:.2f}s")
        
        if failed:
            print(f"\n❌ Failed operations:")
            for result in failed:
                print(f"   {result.asset_symbol} / {result.template_type} / {result.model_name or 'template'}: {result.error}")

def main():
    parser = argparse.ArgumentParser(
        description="DYOR Template & LLM Comparison System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_prompt_template.py --full-pipeline
  python test_prompt_template.py --templates-only --asset RENEW
  python test_prompt_template.py --llm-only --models gpt-4o,claude-3-5-sonnet
  python test_prompt_template.py --compare-reports --show-diff
  python test_prompt_template.py --evaluate-templates
  python test_prompt_template.py --evaluate-templates --asset ECO
        """
    )
    
    # Workflow modes
    parser.add_argument("--simple-test", action="store_true", help="Quick test: process one asset with 2 models (recommended)")
    parser.add_argument("--full-pipeline", action="store_true", help="Run complete pipeline: templates + LLM generation")
    parser.add_argument("--templates-only", action="store_true", help="Only process templates (no LLM calls)")
    parser.add_argument("--llm-only", action="store_true", help="Only run LLM generation (assumes templates exist)")
    parser.add_argument("--compare-reports", action="store_true", help="Compare existing generated reports")
    parser.add_argument("--evaluate-templates", action="store_true", help="AI-powered evaluation: compare templates and rank quality using GPT-4o")
    
    # Filters and options
    parser.add_argument("--asset", help="Process specific asset symbol (e.g., RENEW)")
    parser.add_argument("--template", choices=["old", "new"], help="Process specific template type")
    parser.add_argument("--models", help="Comma-separated list of models to use")
    parser.add_argument("--show-diff", action="store_true", help="Show detailed differences in comparison")
    
    # Utility commands
    parser.add_argument("--list-assets", action="store_true", help="List available assets")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    # Initialize the comparison system
    comparison = DYORComparison()
    
    if args.list_assets:
        assets = comparison.list_available_assets()
        print("📁 Available assets:")
        for asset in assets:
            print(f"   {asset}")
        return
    
    if args.list_models:
        print("🤖 Available models:")
        for model in CONFIG['models']:
            print(f"   {model}")
        return
    
    if args.compare_reports:
        comparison.compare_reports(args.asset, args.show_diff)
        return
    
    if args.evaluate_templates:
        # Check API key for evaluation
        if not comparison.api_key:
            print("❌ No OpenRouter API key found. Template evaluation requires API access.")
            print("   export OPENROUTER_API_KEY='your_key_here'")
            return
        
        print("🤖 AI-Powered Template Evaluation")
        print("=" * 40)
        print(f"Evaluator: {CONFIG['evaluator_model']}")
        print("This will:")
        print("1. Compare old vs new templates for each asset+model pair")
        print("2. Score each report's overall quality (1-10 scale)")
        print("3. Generate comprehensive ranking and analysis")
        print()
        
        results = asyncio.run(comparison.run_template_evaluation(args.asset))
        
        if results:
            print(f"\n🎯 EVALUATION RESULTS SUMMARY")
            print("=" * 50)
            
            template_comps = results.get('template_comparisons', [])
            quality_evals = results.get('quality_evaluations', [])
            
            if template_comps:
                winners = {'OLD': 0, 'NEW': 0, 'TIE': 0}
                for comp in template_comps:
                    winners[comp['winner']] += 1
                
                print(f"📊 Template Comparison Winners:")
                for winner, count in winners.items():
                    percentage = count/len(template_comps)*100 if template_comps else 0
                    print(f"   {winner}: {count} wins ({percentage:.1f}%)")
            
            if quality_evals:
                sorted_evals = sorted(quality_evals, key=lambda x: x.get('normalized_score', 0), reverse=True)
                best_report = sorted_evals[0] if sorted_evals else None
                
                if best_report:
                    print(f"\n🏆 Highest Quality Report:")
                    print(f"   {best_report['asset']} - {best_report['template']} template - {best_report['model']}")
                    print(f"   Score: {best_report.get('normalized_score', 0):.1f}/10")
            
            files = results.get('files', {})
            print(f"\n📁 Detailed results saved:")
            if 'comparisons' in files:
                print(f"   Template comparisons: {files['comparisons']}")
            if 'quality' in files:
                print(f"   Quality evaluations: {files['quality']}")
                
        return
    
    # Handle simple test mode
    if args.simple_test:
        print("🚀 Running Simple Test Mode")
        print("=" * 30)
        print("This will:")
        print("1. Process one asset with both templates")
        print("2. Generate reports with 2 models (to control costs)")
        print("3. Show comparison results")
        print()
        
        # Set defaults for simple test
        if not args.models:
            args.models = "openai/gpt-4o,anthropic/claude-3-5-sonnet-20241022"
        if not args.asset:
            # Pick first available asset
            assets = comparison.list_available_assets()
            if assets:
                args.asset = assets[0]
                print(f"📄 Using asset: {args.asset}")
        
        args.full_pipeline = True
    
    # Parse models
    models = CONFIG['models']
    if args.models:
        models = [m.strip() for m in args.models.split(',')]
    
    print("🚀 DYOR Template & LLM Comparison System")
    print("=" * 50)
    
    all_results = []
    
    # Execute based on mode
    if args.full_pipeline or args.templates_only:
        # Process templates
        template_results = comparison.process_templates(args.asset)
        all_results.extend(template_results)
        
        if args.templates_only:
            comparison.print_summary(template_results)
            return
    
    if args.full_pipeline or args.llm_only:
        # Check API key for LLM operations
        if not comparison.api_key:
            print("❌ No OpenRouter API key found. Set OPENROUTER_API_KEY environment variable.")
            print("   export OPENROUTER_API_KEY='your_key_here'")
            return
        
        # Process LLM generation
        print(f"🤖 Using models: {', '.join(models)}")
        llm_results = asyncio.run(comparison.process_llm_generation(models, args.template, args.asset))
        all_results.extend(llm_results)
    
    # Print final summary
    if all_results:
        comparison.print_summary(all_results)
    
    # Auto-compare for simple test
    if args.simple_test and any(r.success for r in all_results):
        print("\n" + "="*50)
        print("🎯 SIMPLE TEST RESULTS")
        print("="*50)
        comparison.compare_reports(args.asset, True)
        print(f"\n📂 Generated files:")
        print(f"   Templates: {CONFIG['paths']['template_filled']}")
        print(f"   LLM Reports: {CONFIG['paths']['llm_output']}")
        print(f"\n✅ Test complete! Compare the files above to choose the best template + model combination.")
    else:
        print(f"\n📁 Check output directories:")
        print(f"   Templates: {CONFIG['paths']['template_filled']}")
        print(f"   LLM Reports: {CONFIG['paths']['llm_output']}")

if __name__ == "__main__":
    main()
