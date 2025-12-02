
sudo apt-get update
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.10 python3.10-venv python3.10-dev
sudo apt-get install git
sudo apt-get install swig

git clone https://github.com/COMPAT-RL/Representation-Augmented_RL_for_Unified_Navigation-Compat_Agents_in_ViZDoom.git
python3.10 -m venv rl_env
source rl_env/bin/activate
cd Representation-Augmented_RL_for_Unified_Navigation-Compat_Agents_in_ViZDoom
git checkout emanuel_dev
git pull
pip install -r requirements.txt
