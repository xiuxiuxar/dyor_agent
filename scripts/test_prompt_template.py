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
        'anthropic/claude-3-5-sonnet-20241022'
    ],
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
        """Get OpenRouter API key from environment or .env file"""
        # Try environment variable first
        api_key = os.getenv('OPENROUTER_API_KEY')
        if api_key:
            return api_key
            
        # Try .env file
        env_file = Path('.env')
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith('OPENROUTER_API_KEY='):
                        return line.split('=', 1)[1].strip()
        
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
        """
    )
    
    # Workflow modes
    parser.add_argument("--simple-test", action="store_true", help="Quick test: process one asset with 2 models (recommended)")
    parser.add_argument("--full-pipeline", action="store_true", help="Run complete pipeline: templates + LLM generation")
    parser.add_argument("--templates-only", action="store_true", help="Only process templates (no LLM calls)")
    parser.add_argument("--llm-only", action="store_true", help="Only run LLM generation (assumes templates exist)")
    parser.add_argument("--compare-reports", action="store_true", help="Compare existing generated reports")
    
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
