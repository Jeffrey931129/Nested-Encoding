import os

# Target file path
file_path = r"c:\Users\Jeffrey\Downloads\eeg_encoding\04_synthesizing_eeg_data\end_to_end_encoding.py"

print(f"Processing file: {file_path}")

try:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        # Remove trailing newline for processing
        line_content = line.rstrip('\n')
        if not line_content:
            new_lines.append('')
            continue
        
        # Convert existing tabs to spaces first (assuming 4 spaces per tab width for expansion)
        # This handles cases where tabs and spaces are mixed on the same line or block
        expanded = line_content.expandtabs(4)
        
        # Count leading spaces
        stripped = expanded.lstrip(' ')
        if not stripped: # Empty line with whitespace only
            new_lines.append('')
            continue
            
        leading_spaces = len(expanded) - len(stripped)
        
        # Convert to tabs (1 tab = 4 spaces)
        num_tabs = leading_spaces // 4
        remainder_spaces = leading_spaces % 4
        
        # Construct new indentation: primarily tabs, remainder as spaces (though usually 0 if strictly 4-space indent)
        new_indent = '\t' * num_tabs + ' ' * remainder_spaces
        
        new_lines.append(new_indent + stripped)

    # Write back to file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write('\n'.join(new_lines) + '\n')

    print("Indentation fixed successfully: All indentation converted to Tabs (width 4).")

except Exception as e:
    print(f"An error occurred: {e}")
