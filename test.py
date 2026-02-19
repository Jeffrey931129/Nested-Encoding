import os

# Target file path
file_path = r"c:\Users\Jeffrey\Downloads\eeg_encoding\04_synthesizing_eeg_data\end_to_end_encoding.py"

# Read the file
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace tabs with 4 spaces
fixed_content = content.replace('\t', '    ')

# Write the data back
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print(f"Successfully replaced tabs with spaces in: {file_path}")
