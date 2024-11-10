import subprocess

def run_streamlit():
    subprocess.run(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    run_streamlit()
