import subprocess

def run_streamlit():
    subprocess.run(["streamlit", "run", "streamlit.py"])

if __name__ == "__main__":
    run_streamlit()
