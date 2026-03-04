import xml.etree.ElementTree as ET
import os

def extract_event_timestamps(parsed_dir="parsed_inputs", result_dir="timestamp-output"):
    os.makedirs(result_dir, exist_ok=True)
    
    if not os.path.exists(parsed_dir):
        print(f"Error: The folder '{parsed_dir}' does not exist.")
        return

    parsed_files = [f for f in os.listdir(parsed_dir) if f.endswith('.xml')]
    print(f"Found {len(parsed_files)} files in '{parsed_dir}'. Extracting boundaries...")

    processed_count = 0

    for xml_filename in parsed_files:
        xml_path = os.path.join(parsed_dir, xml_filename)
        
        base_name = xml_filename.replace('_parsed.xml', '').replace('.xml', '')
        out_filename = f"{base_name}.time.txt"
        out_path = os.path.join(result_dir, out_filename)
        
        extracted_count = 0
        
        try:
            with open(out_path, 'w') as out_f:
                context = ET.iterparse(xml_path, events=('start',))
                looking_for_timestamp = False
                
                # We put the iteration inside its own try/except block
                # to catch incomplete XML files gracefully.
                try:
                    for event_type, elem in context:
                        if elem.tag == 'event':
                            looking_for_timestamp = True
                        
                        elif looking_for_timestamp:
                            ts = elem.get('timestamp')
                            if ts:
                                out_f.write(f"{ts}\n")
                                extracted_count += 1
                                looking_for_timestamp = False
                        
                        elem.clear()
                        
                except ET.ParseError as pe:
                    # The script hits the end of an incomplete file, catches the error,
                    # and keeps the timestamps it already successfully extracted.
                    print(f"    [!] Note: '{xml_filename}' ended abruptly ({pe}). Recovered {extracted_count} timestamps.")
            
            print(f"  > Success: Extracted {extracted_count} timestamps into '{out_filename}'")
            processed_count += 1
            
        except Exception as e:
            # Catches other non-parsing errors (like permission issues)
            print(f"  ! Error processing '{xml_filename}': {e}")

    print(f"\nExtraction complete! Processed {processed_count} files.")

if __name__ == "__main__":
    extract_event_timestamps()