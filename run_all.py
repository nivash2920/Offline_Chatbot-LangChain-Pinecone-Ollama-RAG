# run_all.py
import subprocess

# List of Python files to run in order
scripts = ["ingestion.py", "retrieval.py", "chatbot_rag.py"]

for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    
    # Print output and errors
    print(result.stdout)
    if result.stderr:
        print(f"Errors in {script}:\n{result.stderr}")

print("All scripts executed successfully.")
