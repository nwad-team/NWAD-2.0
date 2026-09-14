# NWAD-2.0
## How to run

### 1. Clone the repo
git clone https://github.com/oscrjz/NWAD-2.0.git
cd NWAD-2.0

### 2. Create and activate a virtual enviroment 
python -m venv venv 

Activate it:
- Windows: venv\Scripts\Activate.ps1
 (only run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned if theres an error)
 - Mac/Linux: source venv/bin/activate

 ### 3. Install dependencies
 pip install -r requirements.txt

 ### 4. Project Structure
 - src/ - main pipeline code (data prep, model training, evaluation)
 - src/legacy/ - original scapy-based prototype that this project builds on
 - notebooks/ - exploration and experiments (Jupyter)
 - results/ - saved metrics, plots, evaluation outputs
 - data/ - dataset files go here

 ### 5. Dataset 
 Dataset wont be stored in this repo because it is too large for git, once dataset is chosen it will be placed in the data/ folder before running anything that needs it 

 ### 6. Running the code


 ### If you would like to run attackdetect.py
 attackdetect.py requires elevted permission to sniff network traffic, it is not part of the required pipeline as it is kept as a reference demo. But if you would to run it then:
- Windows: install Npcap (npcap.com),then run from an elevated (Admin) termnial 
- Mac/Linux: run with sudo
