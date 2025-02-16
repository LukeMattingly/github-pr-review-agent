from smolagents import CodeAgent,DuckDuckGoSearchTool, HfApiModel,load_tool,tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
from duckduckgo_search import DDGS
import re
import ast



from Gradio_UI import GradioUI

@tool
def search_my_code(github_url:str, code:str)-> str:
    """Searches for code in the github repo given it's url using DuckDuckGo.
    
    Args:
        github_url: The URL of the GitHub repository where the code resides. (e.g., 'https://github.com/LukeMattingly/huggingface-agents-course', 'https://github.com/upb-lea/reinforcement_learning_course_materials').
        code: The new code content to update in the repository.
    
    Returns:
        A string with top search results for code given a github url.
    """
    try:
        query = f"{code} site:{github_url}"
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))  # Get top 5 results

        if results:
            response = "\n".join([f"{res['title']}: {res['href']}" for res in results])
            return f"Here is some code from the github repo {github_url}:\n\n{response}"
        else:
            return f"No results found for {github_url}."

    except Exception as e:
        return f"Error searching for code in {github_url}: {str(e)}"

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
def get_pr_diff(github_url: str, pr_number: int) -> str:
    """Fetches the code diff of a specific pull request.
    
    Args:
        github_url: The URL of the GitHub repository where the pull request is located.
        pr_number: The pull request number for which the code diff should be retrieved.
    
    Returns:
        A string containing the code diff of the specified pull request.
        If the diff cannot be retrieved, returns an error message.
    """
    try:
        owner_repo = github_url.replace("https://github.com/", "")
        api_url = f"https://api.github.com/repos/{owner_repo}/pulls/{pr_number}"
        response = requests.get(api_url, headers={"Accept": "application/vnd.github.v3.diff"})
        
        if response.status_code != 200:
            return f"Error fetching PR diff: {response.json().get('message', 'Unknown error')}"
        
        return response.text[:1000]  # Limit output to avoid overload

    except Exception as e:
        return f"Error retrieving PR diff: {str(e)}"
    

@tool
def get_pr_files_changed(github_url: str, pr_number: int) -> str:
    """Retrieves the list of files changed in a given pull request.
    
    Args:
        github_url: The URL of the GitHub repository where the pull request is located.
        pr_number: The pull request number for which the changed files should be retrieved.
    
    Returns:
        A string listing the files modified in the specified pull request.
        If no files were found or an error occurs, returns an appropriate message.
    """
    try:
        owner_repo = github_url.replace("https://github.com/", "")
        api_url = f"https://api.github.com/repos/{owner_repo}/pulls/{pr_number}/files"
        response = requests.get(api_url)
        
        if response.status_code != 200:
            return f"Error fetching PR files: {response.json().get('message', 'Unknown error')}"
        
        files = response.json()
        return "\n".join([file['filename'] for file in files])

    except Exception as e:
        return f"Error retrieving files for PR #{pr_number}: {str(e)}"

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

final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
custom_role_conversions=None,
)


with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer, get_open_pull_requests, find_todo_comments, get_pr_diff, get_pr_files_changed, detect_code_smells, get_file_content ], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()