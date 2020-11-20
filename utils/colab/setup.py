import os
import subprocess

def setup():
    print('Downloading github repository...')
    os.system('git clone https://github.com/the-dharma-bum/FetalCPSeg/')
    os.chdir('FetalCPSeg')
    os.system('git checkout -b rewrite_network')
    os.system('git branch --set-upstream-to=origin/rewrite_network rewrite_network')
    os.system('git pull -q')
    print('Downloading requirements...')
    os.system('pip install -q -r requirements.txt')


def get_data():
    print("Downloading dataset ...")
    os.system('apt install jq pv')
    os.system('chmod 755 /content/FetalCPSeg/utils/colab/download.sh')
    subprocess.check_call(
        ['/content/FetalCPSeg/utils/colab/download.sh', 
        'https://mega.nz/#!tFNGkLQS!mpq8s6gK2SH6xJOBeYsw62yQlZAN9of4_nHnMjQjfMQ'])
    print("Extracting dataset...")
    os.system('unzip -q patnum_data.zip')