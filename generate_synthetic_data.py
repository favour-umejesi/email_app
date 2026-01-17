#!/usr/bin/env python3
"""
Synthetic Dataset Generator

Generates synthetic emails using gpt-4.1 and appends them to existing datasets.

Usage:
    python generate_synthetic_data.py --task shorten --topic "IT Support" --persona "IT Support Engineer" --tone professional --length medium --count 20
"""

import argparse
import json
import os
from generate import GenerateEmail

def get_next_id(dataset_file: str) -> int:
    """Get the next available ID from a dataset file"""
    max_id = 0
    if os.path.exists(dataset_file):
        with open(dataset_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get('id', 0) > max_id:
                        max_id = data['id']
                except json.JSONDecodeError:
                    continue
    return max_id + 1

def append_to_dataset(dataset_file: str, email_data: dict):
    """Append a single email to the dataset file"""
    with open(dataset_file, 'a') as f:
        f.write(json.dumps(email_data) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic email data')
    parser.add_argument('--task', required=True, choices=['shorten', 'lengthen', 'tone'],
                        help='Task type: shorten, lengthen, or tone')
    parser.add_argument('--topic', required=True,
                        help='Email topic (e.g., "IT Support Request", "HR Onboarding")')
    parser.add_argument('--persona', required=True,
                        help='Sender persona (e.g., "IT Support Engineer", "HR Manager")')
    parser.add_argument('--tone', required=True, choices=['professional', 'friendly', 'sympathetic'],
                        help='Email tone')
    parser.add_argument('--length', required=True, choices=['short', 'medium', 'long'],
                        help='Email length')
    parser.add_argument('--count', type=int, default=20,
                        help='Number of emails to generate (default: 20)')
    
    args = parser.parse_args()
    
    # Determine dataset file - save to separate synthetic files
    dataset_file = f'datasets/{args.task}_synthetic.jsonl'
    
    print(f"\n{'='*60}")
    print(f"Synthetic Email Generator")
    print(f"{'='*60}")
    print(f"Task Type: {args.task}")
    print(f"Topic: {args.topic}")
    print(f"Persona: {args.persona}")
    print(f"Tone: {args.tone}")
    print(f"Length: {args.length}")
    print(f"Count: {args.count}")
    print(f"Dataset: {dataset_file}")
    print(f"{'='*60}\n")
    
    # Initialize generator with gpt-4.1
    generator = GenerateEmail(model="gpt-4.1")
    
    # Get starting ID
    start_id = get_next_id(dataset_file)
    print(f"Starting ID: {start_id}\n")
    
    successful = 0
    failed = 0
    
    for i in range(args.count):
        current_id = start_id + i
        print(f"Generating email {i+1}/{args.count} (ID: {current_id})...", end=" ")
        
        try:
            email_data = generator.generate_synthetic(
                id=current_id,
                task_type=args.task,
                topic=args.topic,
                persona=args.persona,
                tone=args.tone,
                length=args.length
            )
            
            if 'error' in email_data:
                print(f"FAILED - {email_data['error']}")
                failed += 1
            else:
                # Ensure ID is set correctly
                email_data['id'] = current_id
                append_to_dataset(dataset_file, email_data)
                print(f"OK - Subject: {email_data.get('subject', 'N/A')[:40]}...")
                successful += 1
                
        except Exception as e:
            print(f"ERROR - {str(e)}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Generation Complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Dataset updated: {dataset_file}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

