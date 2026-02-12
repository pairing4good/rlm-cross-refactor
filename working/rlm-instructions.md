# RLM Instructions: Find Bob

## Objective
Search through all `.txt` files in the current directory to find which file contains the word "bob" (case-insensitive).

## Task
1. List all `.txt` files in the working directory
2. Search each file for the word "bob"
3. Return the filename of the file that contains "bob"

## Expected Output
Provide the name of the file containing "bob" in the format:
```
The file containing 'bob' is: <filename>
```

## Notes
- Search should be case-insensitive
- Only one file should contain the word "bob"
- Return only the filename, not the full path