#!/usr/bin/env python3
"""
DYOR Agent Prompt Template Tester
=================================

Test script for prompt templates with JSON data. Located in /scripts folder, 
handles full JSON structure with report_data_json and supports batch processing.

DIRECTORY STRUCTURE:
-------------------
scripts/
├── test_prompt_template.py          # This script
├── report_data_jsons/              # Input JSON files (put your data here)
│   ├── sample_renew_report.json    # Sample file 1
│   └── ecotoken_report.json        # Sample file 2
└── report_data_jsons_output/       # Generated prompts (auto-created)
    ├── sample_renew_report_output.txt
    └── ecotoken_report_output.txt

USAGE EXAMPLES:
--------------

1. List Available JSON Files:
   python test_prompt_template.py --list-jsons

2. Process Single JSON File:
   python test_prompt_template.py --json sample_renew_report.json
   python test_prompt_template.py --json sample_renew_report.json --show-output
   python test_prompt_template.py --json sample_renew_report.json --output my_prompt.txt

3. Batch Process All JSON Files:
   python test_prompt_template.py --batch
   python test_prompt_template.py --batch --show-output

4. Use Different Template:
   python test_prompt_template.py --json sample_renew_report.json \
     --template ../packages/xiuxiuxar/skills/dyor_app/prompt_template.jinja

JSON FILE FORMAT:
----------------
Your JSON files should contain the full structure with 'report_data_json':

{
  "asset_id": 35,
  "trigger_id": 65,
  "report_data_json": {
    "asset_info": { "name": "TokenName", "symbol": "TKN", ... },
    "key_metrics": { "price_change_24h": 5.2, ... },
    "social_summary": { "sentiment_score": 88.0, ... },
    "topic_summary": { "overview": "Token description...", ... },
    ...
  },
  "llm_model_used": "Meta-Llama-3-3-70B-Instruct",
  ...
}

The script automatically extracts 'report_data_json' and processes it.

OUTPUT:
-------
- Single file: Creates <filename>_output.txt in current directory
- Batch mode: Creates report_data_jsons_output/ folder with all outputs
- Each output file contains the complete LLM prompt ready to send to an AI model

WORKFLOW:
--------
1. Add your JSON files to scripts/report_data_jsons/
2. Run: python test_prompt_template.py --batch
3. Find generated prompts in scripts/report_data_jsons_output/
4. Use the generated prompts with your LLM API calls
"""

import json
import sys
import argparse
from datetime import datetime
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

def timestamp_to_date(timestamp_str):
    """Convert timestamp string to readable date format"""
    try:
        if isinstance(timestamp_str, (int, float)):
            # Unix timestamp
            dt = datetime.fromtimestamp(timestamp_str)
        else:
            # ISO format string
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%B %d, %Y")
    except:
        return str(timestamp_str)

def intcomma(value):
    """Add commas to numbers for readability"""
    try:
        return f"{float(value):,.0f}"
    except:
        return str(value)

def load_json_data(json_path):
    """Load JSON data from file and extract report_data_json if present"""
    try:
        with open(json_path, 'r') as f:
            full_data = json.load(f)
        
        # Extract the actual report data from the full structure
        if "report_data_json" in full_data:
            data = full_data["report_data_json"]
            print(f"   ✓ Extracted report_data_json from {json_path.name}")
        else:
            # Assume it's already the report data structure
            data = full_data
            print(f"   ✓ Using {json_path.name} as direct report data")
        
        # Convert topic_summary dict to string if needed
        if "topic_summary" in data and isinstance(data["topic_summary"], dict):
            data["topic_summary"] = data["topic_summary"].get("overview", "")
            
        return data
    except Exception as e:
        print(f"   ❌ Error loading {json_path.name}: {e}")
        return None

def render_template(template_path, data, output_file=None):
    """Render a template with given data"""
    # Check if template file exists
    if not template_path.exists():
        print(f"❌ Template file not found at {template_path}")
        return None

    # Set up Jinja2 environment
    template_dir = template_path.parent
    env = Environment(loader=FileSystemLoader(template_dir))
    
    # Add custom filters
    env.filters['timestamp_to_date'] = timestamp_to_date
    env.filters['intcomma'] = intcomma
    
    # Load the template
    try:
        template = env.get_template(template_path.name)
    except Exception as e:
        print(f"❌ Error loading template: {e}")
        return None

    # Render the template
    try:
        rendered = template.render(**data)
        
        if output_file:
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(rendered)
        
        return rendered
        
    except Exception as e:
        print(f"❌ Error rendering template: {e}")
        import traceback
        traceback.print_exc()
        return None

def list_available_jsons():
    """List available JSON files in report_data_jsons folder"""
    json_dir = Path("report_data_jsons")
    if not json_dir.exists():
        print("❌ report_data_jsons folder not found")
        return []
    
    json_files = list(json_dir.glob("*.json"))
    if json_files:
        print(f"📁 Available JSON files in {json_dir}:")
        for i, json_file in enumerate(json_files, 1):
            print(f"   {i}. {json_file.name}")
    else:
        print(f"📁 No JSON files found in {json_dir}")
    
    return json_files

def process_single_file(json_file, template_path, output_dir=None, show_output=False):
    """Process a single JSON file"""
    data = load_json_data(json_file)
    if not data:
        return False
    
    # Determine output file name
    if output_dir:
        output_file = output_dir / f"{json_file.stem}_output.txt"
    else:
        output_file = Path(f"{json_file.stem}_output.txt")
    
    rendered = render_template(template_path, data, output_file)
    
    if rendered:
        print(f"   ✓ Generated: {output_file}")
        if show_output:
            print(f"\n--- OUTPUT for {json_file.name} ---")
            print(rendered[:500] + "..." if len(rendered) > 500 else rendered)
            print("--- END OUTPUT ---\n")
        return True
    return False

def batch_process_jsons(template_path, show_output=False):
    """Process all JSON files in report_data_jsons folder"""
    json_dir = Path("report_data_jsons")
    output_dir = Path("report_data_jsons_output")
    
    if not json_dir.exists():
        print("❌ report_data_jsons folder not found")
        return 0
    
    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        print("❌ No JSON files found in report_data_jsons folder")
        return 0
    
    print(f"🚀 Batch processing {len(json_files)} JSON files...")
    print(f"📁 Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    success_count = 0
    for json_file in json_files:
        print(f"\n�� Processing: {json_file.name}")
        if process_single_file(json_file, template_path, output_dir, show_output):
            success_count += 1
    
    print(f"\n✅ Batch complete! {success_count}/{len(json_files)} files processed successfully")
    print(f"📁 Check {output_dir} for output files")
    
    return success_count

def main():
    parser = argparse.ArgumentParser(
        description="Test prompt templates with JSON data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_prompt_template.py --list-jsons
  python test_prompt_template.py --json sample_renew_report.json
  python test_prompt_template.py --batch
  python test_prompt_template.py --batch --show-output
        """
    )
    parser.add_argument("--template", default="../packages/xiuxiuxar/skills/dyor_app/prompt_template_updated.jinja", 
                       help="Path to Jinja2 template file (relative to scripts folder)")
    parser.add_argument("--json", help="Path to JSON data file (can be relative to report_data_jsons/)")
    parser.add_argument("--output", help="Output file path (default: <json_name>_output.txt)")
    parser.add_argument("--batch", action="store_true", help="Process all JSON files in report_data_jsons folder")
    parser.add_argument("--show-output", action="store_true", help="Print rendered output to console")
    parser.add_argument("--list-jsons", action="store_true", help="List available JSON files")
    
    args = parser.parse_args()
    
    if args.list_jsons:
        list_available_jsons()
        return
    
    print("🚀 DYOR Agent Prompt Template Tester")
    print("=" * 50)
    
    # Load template
    template_path = Path(args.template)
    if not template_path.exists():
        print(f"❌ Template not found: {template_path}")
        sys.exit(1)
    
    print(f"📄 Using template: {template_path.name}")
    
    if args.batch:
        # Batch process all JSON files
        batch_process_jsons(template_path, args.show_output)
    elif args.json:
        # Process single JSON file
        json_path = Path(args.json)
        if not json_path.exists() and not json_path.is_absolute():
            # Try in report_data_jsons folder
            json_path = Path("report_data_jsons") / args.json
        
        if not json_path.exists():
            print(f"❌ JSON file not found: {args.json}")
            sys.exit(1)
        
        print(f"📄 Processing: {json_path.name}")
        
        # Determine output file
        if args.output:
            output_file = Path(args.output)
        else:
            output_file = Path(f"{json_path.stem}_output.txt")
        
        success = process_single_file(json_path, template_path, 
                                    output_file.parent if args.output else None, 
                                    args.show_output)
        
        if success:
            print(f"\n✅ Complete! Output saved to: {output_file}")
        else:
            print("\n❌ Processing failed")
            sys.exit(1)
    else:
        # No JSON specified, show help
        print("ℹ️  No JSON file specified. Available options:")
        print("   --json <filename>     Process single file")
        print("   --batch              Process all files in report_data_jsons/")
        print("   --list-jsons         Show available JSON files")
        print("\nExample usage:")
        print("   python test_prompt_template.py --json sample_renew_report.json")
        print("   python test_prompt_template.py --batch")

if __name__ == "__main__":
    main()
