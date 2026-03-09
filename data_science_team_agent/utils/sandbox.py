import subprocess
import sys
import tempfile
import os
import pandas as pd
from typing import Tuple, Any, Dict


def run_code_sandboxed_subprocess(
    code_snippet: str,
    function_name: str,
    data: Dict[str, Any],
    timeout: int = 10,
    memory_limit_mb: int = 512,
) -> Tuple[Any, str]:
    """
    Run code in a sandboxed subprocess.
    
    Note: This is a simplified implementation. In production, use proper sandboxing.
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        # Write the code and data loading
        f.write(f"""
import pandas as pd
import numpy as np
from typing import Dict, Any

# Data provided
data = {data}

# User code
{code_snippet}

# Execute the function
if '{function_name}' in locals():
    result = {function_name}(pd.DataFrame(data))
    print("RESULT_START")
    print(result.to_json())
    print("RESULT_END")
else:
    print(f"Function '{function_name}' not found")
""")
        temp_file = f.name
    
    try:
        # Run the code in subprocess
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir()
        )
        
        if result.returncode != 0:
            return None, f"Execution error: {result.stderr}"
        
        # Extract result from output
        output = result.stdout
        result_start = output.find("RESULT_START")
        result_end = output.find("RESULT_END")
        
        if result_start != -1 and result_end != -1:
            json_str = output[result_start + len("RESULT_START"):result_end].strip()
            try:
                import json
                result_data = json.loads(json_str)
                return pd.DataFrame(result_data), None
            except json.JSONDecodeError as e:
                return None, f"JSON decode error: {e}"
        else:
            return None, "No result found in output"
            
    except subprocess.TimeoutExpired:
        return None, f"Execution timed out after {timeout} seconds"
    except Exception as e:
        return None, f"Sandbox error: {e}"
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file)
        except:
            pass


def validate_code_safety(code: str) -> Tuple[bool, str]:
    """Basic code safety validation."""
    dangerous_patterns = [
        r'os\.system',
        r'subprocess\.call',
        r'subprocess\.run',
        r'eval\s*\(',
        r'exec\s*\(',
        r'__import__',
        r'open\s*\(',
        r'file\s*\(',
        r'input\s*\(',
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, code):
            return False, f"Potentially dangerous code detected: {pattern}"
    
    return True, "Code appears safe"


import re
