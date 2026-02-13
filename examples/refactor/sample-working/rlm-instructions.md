# RLM Instructions: Find Bob

## Objective
Search through all `.txt` files across all repositories to find which file contains the word "bob" (case-insensitive).

## Task
1. Explore all git repositories in the working directory
2. Find all `.txt` files across all repositories
3. Search each file for the word "bob"
4. Return the repository and filename that contains "bob"

## Expected Output
Provide the location of the file containing "bob" in the format:
```
The file containing 'bob' is: <repository>/<filename>
```

## Notes
- Search should be case-insensitive
- Search across ALL repositories in the working directory
- Only one file should contain the word "bob"
- Return the relative path from the working directory (e.g., "sample-repo-three/file_05.txt")