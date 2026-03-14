#!/usr/bin/env python3
"""
Streaming-Based Terminal Log Boundary Generator

This script processes raw XML terminal logs and converts them into a sliding-window 
training dataset for LLM fine-tuning. It feeds the model the last N lines of context 
and asks it to classify if the final line is a "new event" (boundary) or an "old event".

Key Features:
- Structural Isolation: Physically separates context from the target line to prevent attention traps.
- Sliding Window Context: Keeps the model's attention focused on the immediate context.
- Exact Ratio Balancing: Guarantees a specific ratio of Old to New events (e.g., 2:1).
- Smart Negative Sampling: Prioritizes "Hard Negatives" (user pressing Enter) to fill the quota.
- String-based Timestamp Matching: Prevents floating-point precision errors.
"""

import os
import re
import json
import random
import argparse

# --- CONFIGURATION ---
DEFAULT_XML = "inputs/1728236638.rec.xml"
DEFAULT_TIME = "timestamp-output/1728236638.time.txt"

WINDOW_SIZE = 15       # Number of XML chunks to show the model at once
NEGATIVE_RATIO = 2     # How many 'old events' to keep for every 1 'new event' (e.g., 2 = 2:1 ratio)

# Regex for extracting XML chunks and timestamps
CHUNK_RE = re.compile(r"<(user_input|system_output)\b[^>]*?(?:/>|>.*?</\1>)", flags=re.DOTALL)
TIME_RE = re.compile(r'timestamp="([\d\.]+)"')

SYSTEM_PROMPT = """Your task is to analyze terminal XML logs and determine whether the timestamp in the TARGET LINE belongs to a "new event" or an "old event".

### DEFINITION OF A NEW EVENT:
1. **Explicit Prompts:** The very first `<user_input>` that immediately follows a shell prompt (e.g., `demo@faiserver:~$`).
2. **Phase Transitions:** In automated logs, moving from one major build stage to another (e.g., from 'fai-mirror finished' to 'Copying the nfsroot').
3. **Internal Logic:** Shifts from downloading to processing.

### WHAT IS *NOT* A NEW EVENT (OLD EVENT):
- **User Input / Keystrokes:** A user typing a command, including pressing the Enter key (a newline `\\n`), is just the completion of the input phase.
- **Incomplete Tasks:** Continuous system output without a clear phase shift.

CRITICAL INSTRUCTION: You must classify ONLY the timestamp found in the "### TARGET LINE" section. Do NOT extract timestamps from the "### CONTEXT" section. Output only the timestamp and the classification. Do NOT use brackets, periods, explanations, or markdown formatting.

Output Format Example 1: 39.229814, old event
Output Format Example 2: 111.602501, new event"""

def get_timestamp(chunk_text: str) -> str:
    """Extracts the timestamp as a string to avoid float precision issues."""
    match = TIME_RE.search(chunk_text)
    return match.group(1) if match else ""

def is_hard_negative(chunk_text: str) -> bool:
    """Identifies if a chunk is a 'Hard Negative' (e.g., a user pressing Enter)."""
    if "<user_input>" in chunk_text and "\n" in chunk_text:
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Create a sliding window dataset for event boundary detection.")
    parser.add_argument("--xml", default=DEFAULT_XML, help="Path to raw XML log.")
    parser.add_argument("--time", default=DEFAULT_TIME, help="Path to ground truth timestamps txt file.")
    parser.add_argument("--out", default="streaming_dataset.jsonl", help="Output JSONL dataset path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for negative sampling.")
    args = parser.parse_args()

    random.seed(args.seed)

    # 1. Load ground truth boundaries into a fast-lookup Set
    if not os.path.exists(args.time):
        print(f"Error: Ground truth file {args.time} not found.")
        return
        
    with open(args.time, 'r') as f:
        boundaries_set = set(line.strip() for line in f if line.strip())

    print(f"DEBUG: Loaded {len(boundaries_set)} boundary timestamps.")

    # 2. Parse raw XML into sequential chunks
    with open(args.xml, 'r') as f:
        xml_content = f.read()
    
    all_chunks = []
    for match in CHUNK_RE.finditer(xml_content):
        raw_text = match.group(0)
        ts_str = get_timestamp(raw_text)
        if ts_str:
            all_chunks.append({"ts": ts_str, "text": raw_text})
            
    print(f"DEBUG: Extracted {len(all_chunks)} valid XML chunks.")

    if len(all_chunks) < WINDOW_SIZE:
        print(f"Error: XML file has fewer chunks ({len(all_chunks)}) than WINDOW_SIZE ({WINDOW_SIZE}).")
        return

    # 3. Sliding Window & Collection Phase
    positives = []
    hard_negatives = []
    easy_negatives = []
    
    for i in range(WINDOW_SIZE, len(all_chunks) + 1):
        window_chunks = all_chunks[i - WINDOW_SIZE : i]
        
        # --- STRUCTURAL ISOLATION LOGIC ---
        context_chunks = window_chunks[:-1]
        target_chunk = window_chunks[-1]
        target_ts = target_chunk["ts"]
        
        is_boundary = target_ts in boundaries_set
        
        context_text = "\n".join([c["text"] for c in context_chunks])
        target_text = target_chunk["text"]
        
        structured_input = f"### CONTEXT (Previous Events):\n{context_text}\n\n### TARGET LINE (Extract and Classify THIS Timestamp):\n{target_text}"
        # ----------------------------------
        
        # Package the data point and keep the index 'i' so we can sort chronologically later
        data_point = {
            "index": i, 
            "json_data": {
                "instruction": SYSTEM_PROMPT,
                "input": structured_input,
                "output": f"{target_ts}, {'new' if is_boundary else 'old'} event"
            }
        }
        
        if is_boundary:
            positives.append(data_point)
        else:
            if is_hard_negative(target_text):
                hard_negatives.append(data_point)
            else:
                easy_negatives.append(data_point)

    # 4. Enforce the Ratio
    num_positives = len(positives)
    target_negatives = num_positives * NEGATIVE_RATIO
    
    selected_negatives = []
    
    # Shuffle the negatives so we grab a random assortment
    random.shuffle(hard_negatives)
    random.shuffle(easy_negatives)

    # Grab hard negatives first
    if len(hard_negatives) >= target_negatives:
        selected_negatives = hard_negatives[:target_negatives]
    else:
        selected_negatives = hard_negatives
        remaining_needed = target_negatives - len(selected_negatives)
        
        # Fill the rest with easy negatives
        if len(easy_negatives) >= remaining_needed:
            selected_negatives += easy_negatives[:remaining_needed]
        else:
            selected_negatives += easy_negatives # Fallback if there just aren't enough chunks

    # 5. Recombine, Sort Chronologically, and Save
    final_dataset = positives + selected_negatives
    # Sort by the original index 'i' so the log flows forward in time
    final_dataset.sort(key=lambda x: x["index"])

    with open(args.out, 'w') as f:
        for item in final_dataset:
            f.write(json.dumps(item["json_data"], ensure_ascii=False) + "\n")
            
    print(f"\n✅ Done! Created {len(final_dataset)} total examples in '{args.out}'.")
    print(f"📊 Dataset Balance: {len(positives)} Positives (New) | {len(selected_negatives)} Negatives (Old)")

if __name__ == "__main__":
    main()