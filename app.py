from smolagents import CodeAgent, HfApiModel,tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
import re
import ast
from typing import List
from huggingface_hub import login
import os


from Gradio_UI import GradioUI


@tool
def get_open_pull_requests(github_url: str) -> str:
    """Fetches a list of open pull requests for a given GitHub repository.
    
    Args:
        github_url: The URL of the GitHub repository where the pull requests should be retrieved. 
                    (e.g., 'https://github.com/LukeMattingly/huggingface-agents-course', 
                    'https://github.com/upb-lea/reinforcement_learning_course_materials').
    
    Returns:
        A string containing the list of open pull requests with their titles and links.
        If no pull requests are open, returns a message indicating no PRs were found.
    """
    try:
        owner_repo = github_url.replace("https://github.com/", "")
        api_url = f"https://api.github.com/repos/{owner_repo}/pulls"
        response = requests.get(api_url)
        
        if response.status_code != 200:
            return f"Error fetching PRs: {response.json().get('message', 'Unknown error')}"
        
        pull_requests = response.json()
        if not pull_requests:
            return "No open pull requests found."
        
        return "\n".join([f"PR #{pr['number']}: {pr['title']} - {pr['html_url']}" for pr in pull_requests])

    except Exception as e:
        return f"Error retrieving pull requests: {str(e)}"

@tool
def find_todo_comments(code: str) -> str:
    """Finds TODO and FIXME comments in the provided code.
    
    Args:
        code: The source code in which to search for TODO and FIXME comments.
    
    Returns:
        A string listing all TODO and FIXME comments found in the code.
        If no comments are found, returns a message indicating that no TODO or FIXME comments exist.
    """
    matches = re.findall(r"#\s*(TODO|FIXME):?\s*(.*)", code, re.IGNORECASE)
    
    if not matches:
        return "No TODO or FIXME comments found."
    
    return "\n".join([f"{match[0]}: {match[1]}" for match in matches])

@tool
def get_pr_diff(github_url: str, pr_number: int, start_line: int = None, end_line: int = None, total_lines: int = None) -> str:
    """Fetches the code diff of a specific pull request and returns a subset of lines as requested.
    
    Args:
        github_url: The URL of the GitHub repository where the pull request is located.
                    (e.g., 'https://github.com/crewAIInc/crewAI').
        pr_number: The pull request number for which the code diff should be retrieved.
        start_line: Optional; the starting line number (1-indexed) of the diff to return.
        end_line: Optional; the ending line number (1-indexed) of the diff to return.
        total_lines: Optional; if provided, returns the first 'total_lines' lines of the diff.
                     This parameter is ignored if both start_line and end_line are provided.
    
    Returns:
        A string containing the requested portion of the code diff of the specified pull request.
        If the diff cannot be retrieved or if invalid parameters are provided, returns an error message.
    """
    try:
        owner_repo = github_url.replace("https://github.com/", "")
        api_url = f"https://api.github.com/repos/{owner_repo}/pulls/{pr_number}"
        response = requests.get(api_url, headers={"Accept": "application/vnd.github.v3.diff"})
        
        if response.status_code != 200:
            return f"Error fetching PR diff: {response.json().get('message', 'Unknown error')}"
        
        diff_text = response.text
        # Split the diff into individual lines
        diff_lines = diff_text.splitlines()
        
        # Determine which subset of lines to return:
        if start_line is not None or end_line is not None:
            if start_line is None or end_line is None:
                return "Error: Both start_line and end_line must be provided if specifying a range."
            # Adjust for 1-indexed line numbers provided by the user.
            diff_lines = diff_lines[start_line - 1:end_line]
        elif total_lines is not None:
            diff_lines = diff_lines[:total_lines]
        
        return "\n".join(diff_lines)
    except Exception as e:
        return f"Error retrieving PR diff: {str(e)}"

@tool
def get_pr_diff_for_file(github_url: str, pr_number: int, file_path: str) -> str:
    """Fetches the code diff for a specific file in a given pull request.
    
    Args:
        github_url: The URL of the GitHub repository where the pull request is located.
                    (e.g., 'https://github.com/crewAIInc/crewAI').
        pr_number: The pull request number for which the diff should be retrieved.
        file_path: The relative path of the file within the repository to retrieve the diff for
                   (e.g., 'src/module.py').
    
    Returns:
        A string containing the code diff (patch) for the specified file in the pull request.
        If the file is not found in the PR or if its diff is not available, returns an error message.
    """
    try:
        # Extract owner and repo from the URL
        owner_repo = github_url.replace("https://github.com/", "")
        # API endpoint to get files changed in the PR
        api_url = f"https://api.github.com/repos/{owner_repo}/pulls/{pr_number}/files"
        response = requests.get(api_url)
        
        if response.status_code != 200:
            return f"Error fetching PR files: {response.json().get('message', 'Unknown error')}"
        
        files = response.json()
        # Look for the specific file in the list
        for file_info in files:
            if file_info.get('filename') == file_path:
                patch = file_info.get('patch')
                if patch:
                    return patch
                else:
                    return f"No diff (patch) available for file: {file_path}"
        
        return f"File '{file_path}' not found in the pull request."
    except Exception as e:
        return f"Error retrieving PR diff for file: {str(e)}"


@tool
def get_pr_files_changed(github_url: str, pr_number: int) -> List[str]:
    """Retrieves the list of files changed in a given pull request.
    
    Args:
        github_url: The URL of the GitHub repository where the pull request is located.
        pr_number: The pull request number for which the changed files should be retrieved.
    
    Returns:
        A list of strings, where each string is a file path that was modified in the specified pull request.
        If no files are found or an error occurs, returns a list with an appropriate error message.
    """
    try:
        owner_repo = github_url.replace("https://github.com/", "")
        api_url = f"https://api.github.com/repos/{owner_repo}/pulls/{pr_number}/files"
        response = requests.get(api_url)
        
        if response.status_code != 200:
            return [f"Error fetching PR files: {response.json().get('message', 'Unknown error')}"]
        
        files = response.json()
        return [file['filename'] for file in files]

    except Exception as e:
        return [f"Error retrieving files for PR #{pr_number}: {str(e)}"]

@tool
def detect_code_smells(code: str) -> str:
    """Detects common code smells such as long functions and deeply nested loops.
    
    Args:
        code: The source code to analyze for potential code smells.
    
    Returns:
        A string listing detected code smells, including long functions and deeply nested loops.
        If no code smells are found, returns a message indicating the code is clean.
    """
    try:
        tree = ast.parse(code)
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and len(node.body) > 20:
                issues.append(f"Long function detected: {node.name} ({len(node.body)} lines)")
            if isinstance(node, ast.For) or isinstance(node, ast.While):
                nested_loops = sum(isinstance(n, (ast.For, ast.While)) for n in ast.walk(node))
                if nested_loops > 2:
                    issues.append(f"Deeply nested loop detected in function: {node.lineno}")

        return "\n".join(issues) if issues else "No code smells detected."

    except Exception as e:
        return f"Error analyzing code: {str(e)}"
'''
@tool
def get_file_content(github_url: str, file_path: str) -> str:
    """Fetches the content of a specific file from the GitHub repository.
    
    Args:
        github_url: The URL of the GitHub repository (e.g., 'https://github.com/user/repo').
        file_path: The relative path of the file within the repository (e.g., 'src/module.py').
    
    Returns:
        A string containing the file's content or an error message if retrieval fails.
    """
    try:
        owner_repo = github_url.replace("https://github.com/", "")
        api_url = f"https://raw.githubusercontent.com/{owner_repo}/main/{file_path}"
        response = requests.get(api_url)
        if response.status_code != 200:
            return f"Error fetching file content: {response.status_code}"
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

        ''' 
@tool
def security_check_code(code: str) -> str:
    """Analyzes the provided code snippet for potential security vulnerabilities.
    
    Args:
        code: The source code to be analyzed for common security issues (e.g., hardcoded secrets, unsafe functions).
    
    Returns:
        A string listing detected potential security vulnerabilities based on common patterns (e.g., hardcoded credentials,
        risky usage of functions like eval or os.system, and simple SQL injection risks). If no issues are found, returns a message indicating the code is secure.
    """
    import re
    issues = []
    
    # Check for hardcoded credentials (case-insensitive search)
    secret_patterns = [
        r'(?i)api[-_]?key\s*=\s*[\'"].+[\'"]',
        r'(?i)secret\s*=\s*[\'"].+[\'"]',
        r'(?i)password\s*=\s*[\'"].+[\'"]',
        r'(?i)token\s*=\s*[\'"].+[\'"]'
    ]
    for pattern in secret_patterns:
        matches = re.findall(pattern, code)
        if matches:
            issues.append("Potential hardcoded credential(s) found: " + ", ".join(matches))
    
    # Check for usage of eval() which can be dangerous
    if "eval(" in code:
        issues.append("Usage of eval() detected, which can lead to security vulnerabilities if misused.")
    
    # Check for potential command injection risks with os.system
    if "os.system(" in code:
        issues.append("Usage of os.system() detected; consider using safer alternatives to avoid command injection risks.")
    
    # Check for simple SQL injection patterns (heuristic)
    sql_injection_patterns = [
        r"execute\(.+\+.+\)",
        r"format\(.+%\(.+\)s.+\)"
    ]
    for pattern in sql_injection_patterns:
        matches = re.findall(pattern, code)
        if matches:
            issues.append("Potential SQL injection risk found in statements: " + ", ".join(matches))
    
    if issues:
        return "\n".join(issues)
    else:
        return "No obvious security vulnerabilities detected based on heuristic analysis."

@tool
def check_documentation_updates(changed_files: str) -> str:
    """Checks whether documentation files have been updated alongside code changes.
    
    Args:
        changed_files: A newline-separated string listing the file paths changed in a commit or pull request.
    
    Returns:
        A string indicating whether documentation appears to have been updated or if it might be missing.
    """
    files = [f.strip() for f in changed_files.splitlines() if f.strip()]
    doc_files = [f for f in files if "readme" in f.lower() or "docs" in f.lower()]
    
    if doc_files:
        return "Documentation files were updated."
    else:
        return "No documentation updates detected. Consider reviewing the docs to ensure they reflect the new changes."

@tool
def lint_code(code: str) -> str:
    """Analyzes the provided code snippet for style and potential issues using a linter.
    
    Args:
        code: The source code to be analyzed.
    
    Returns:
        A string with linting warnings and suggestions for improvement, or a message indicating that no issues were found.
    """
    # This is a placeholder; you could integrate pylint or flake8 via subprocess or an API.
    # For demonstration, we'll simulate a response.
    issues = []
    if "print(" in code:
        issues.append("Consider removing debug print statements.")
    if not issues:
        return "No linting issues found."
    return "\n".join(issues)


final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded deepseek-ai/DeepSeek-R1-Distill-Qwen-32B || Qwen/Qwen2.5-Coder-32B-Instruct
custom_role_conversions=None,
)


with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer, get_open_pull_requests, find_todo_comments, get_pr_diff, get_pr_files_changed, detect_code_smells, security_check_code, check_documentation_updates, lint_code, get_pr_diff_for_file ], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()