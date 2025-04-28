### 1. Creating a Virtual Machine in Azure Cloud

1. Navigate to Dashboard → Virtual Machine → Create Virtual Machine.
2. Select the Azure for Students subscription and create a new resource group named **WOA-RG**.
3. Enter a custom virtual machine name, such as **woa-experiments-instance-01**.
4. Select **(Europe) West Europe** as the region and **Zone 3** as the availability zone.
5. Select **F4s_v2** from F-series v2 as virtual machine size. Price ~ 141,62 USD/month for 4vCPUS and 8GB of RAM.
6. Choose Ubuntu Server 24.04 LTS - x64 Gen2 as the base image.
7. Keep the username as **azureuser**.
8. Change the key pair name to **azureuserkey**.
9. Keep SSH public key as the default authentication option. Generate a new key pair using the RSA SSH format.
10. Leave all inbound port rules at their default settings.
11. Click Review + Create. Then Create when validation passed.
12. Make sure to download key and create resources. Complete deployment may take up to 2-5 minutes.
13. Change SSH key permissions `chmod 600 ~/Downloads/azureuserkey.pem`

### 2. Cost Management
1. Make sure to stop the virtual machine to avoid unnecessary charges when not used.
2. Upon completing the experiments, ensure that all resources within the created resource group are deleted.

### 3. Connect to Virtual Machine using SSH in Visual Studio Code
1. Download **Remote Explorer** and **Remote - SSH** extensions in VSCode.
2. In Azure Portal, go to Virtual Machines ->  **woa-experiments-instance-01** -> Connect -> connect.
3. Select Native SSH and copy the connection command such as `ssh -i ~/Downloads/azureuserkey.pem
   azureuser@<IP_ADDRESS>`
4. In VSCode under **Remote Explorer** plugin, add new connection typing the above ssh command.
5. Connect to the new host and wait until configuration completed.

### 4. Configure Virtual Machine, setup environment and run experimental suite
1. sudo apt update
2. sudo apt install build-essential
3. sudo apt install python3-dev
4. Clone repository
5. git config --global user.name "<GIT_NAME_SURNAME>"
6. git config --global user.email "<GIT_EMAIL_ADDRESS>"
7. ssh-keygen -t ed25519 -C "<GIT_EMAIL_ADDRESS>"
8. cat /home/azureuser/.ssh/id_ed25519.pub
9. Add the above public key to your GitHub account.
10. curl -LsSf https://astral.sh/uv/install.sh | sh
11. source $HOME/.local/bin/env
12. uv venv .venv
13. source .venv/bin/activate
14. uv pip install -r pyproject.toml
15. Your Virtual Machine is ready to run experiments
16. Install all pre-commit plugins `pre-commit install`
17. Execute pre-commit `pre-commit run --all-files`
18. Make sure to run experiments using either `tmux` or `nohup`
