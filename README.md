# humoid-gui


# One Love IPFS Humoid GUI Release V1.0.0

![image](https://github.com/graylan0/humoid-gui/assets/34530588/7fd5a62d-02a5-4d0e-b33f-edb04d923f55)





We are thrilled to announce the first full release of Humoid GUI, version 1.0.0! This release introduces a fully functional graphical user interface application for interacting with humanoid robots or simulations. Humoid GUI combines AI, database integration, and natural language processing to provide an interactive and user-friendly experience.

## AI Features

- **AI-Powered Conversations**: Dynamic conversations with AI, powered by the Llama natural language processing model.
- **Database Integration**: Interactions stored in a Vector database for persistence and analysis.
- **Image Generation**: Generate images based on user inputs, adding a visual dimension to the interaction experience.


## Installation

Humoid GUI is now available as an executable file for Windows. Download the latest version from our [releases page](https://github.com/graylan0/humoid-gui/releases) and run the installer.

## System Requirements

- Compatible with Windows 10 and later.
- Minimum 32GB RAM recommended.
- Minimum Nvidia GPU with 12GB VRAM (Tested with 20GB RTX A4500)
- At least 20GB of free disk space.

## Usage

After downloading the all the 7zip packages part by part and extracting with them all with 7zip, you can start Humoid GUI exe (dave_v1.exe) from your desktop or start menu by creating a shortcut.


## Future Plans

- Integration with additional humanoid robot APIs.
- Enhanced customization options for the GUI.
- Further improvements to AI algorithms for more natural interactions.
- Plans to support macOS and Linux in future releases.



Thank you contributors and members of the freedomdao community who made this project possible.


![image](https://github.com/graylan0/humoid-gui/assets/34530588/7118014e-0698-4c11-9407-1c08fe3fd31a)



![image](https://github.com/graylan0/humoid-gui/assets/34530588/b9644ccf-13f0-4600-bfad-b9a45ba5017c)




# Weaviate Vector Database tutorial



To boot up the Weaviate components using the provided `docker-compose.yml` file and set up backups with the `vecbackup` extension, follow these steps:

### Step 1: Booting Up Weaviate with Docker Compose

1. **Ensure Docker and Docker Compose are Installed**: Make sure you have Docker and Docker Compose installed on your system.

2. **Download the `docker-compose.yml` File**: Clone the repository or directly download the `docker-compose.yml` file from [here](https://github.com/graylan0/humoid-gui/blob/main/docker-compose.yml).

3. **Navigate to the Directory**: Open a terminal and navigate to the directory where the `docker-compose.yml` file is located.

4. **Run Docker Compose**: Execute the following command to start the Weaviate services:

   ```bash
   docker-compose up -d
   ```

   This command will start all the services defined in your `docker-compose.yml`, including Weaviate and its dependencies like `contextionary`, `qna-transformers`, `ner-transformers`, and `sum-transformers`.

### Step 2: Setting Up Backups

1. **Download the Backup File**: Download the `weave_weaviate_data.tar.tar.001` files part by part from [here](https://github.com/graylan0/ModeZion/blob/main/vecbackup/weave_weaviate_data.tar.tar.001).

2. **Extract the Backup**: Use a tool like `tar` to extract the backup file. If the backup is split into multiple parts, you may need to combine them before extraction.

   ```bash
   tar -xvf weave_weaviate_data.tar.tar.001
   ```

3. **Place the Backup in a Volume**: The extracted data should be placed in a volume that your Weaviate container will use. According to your `docker-compose.yml`, this volume is named `weaviate_data`.

4. **Configure the Volume**: Ensure that the `docker-compose.yml` file is configured to use the volume where you've placed the backup. It should look something like this:

   ```yaml
   volumes:
     weaviate_data:
       external: true
   ```

   You might need to create a Docker volume that points to the location of your backup data.

### Step 3: Restart Weaviate

After setting up the backup data in the correct volume, restart the Weaviate services:

```bash
docker-compose down
docker-compose up -d
```

This will ensure that Weaviate starts with the data from your backup.

### Additional Notes:

- **Volume Management**: If you're not familiar with Docker volumes, you might need to read up on how to manage data in Docker volumes.
- **Backup Integrity**: Ensure that the backup data is not corrupted and is compatible with the version of Weaviate you are running.
- **Weaviate Configuration**: Double-check the Weaviate configuration in your `docker-compose.yml` file to ensure it aligns with your backup and intended setup.

By following these steps, you should be able to boot up Weaviate with the necessary configurations and data backups for your Humoid GUI application.


# Automatic1111 Image Model Tutorial Url 


Note: this is for building the image model (API running on localhost or the IP/url you select)
1. Install Automatic1111  `sd.webui.zip` from `v1.0.0-pre here`[here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) by extracting the foldering inside zip file. (Nvidia Windows 10+ setup)
2 Start Automatic1111 by running `run.bat` . This should start the Automatic1111 WebUI and load the model to be used with the GUI program.
3. Use Invoke's built in VAE Selected within Automatic1111's settings. Install Deliberate for invoke 08 model from [here](https://civitai.com/models/5585/deliberate-for-invoke) .


![image](https://github.com/graylan0/ModeZion/assets/34530588/3d0c4be9-61ca-4936-9216-11b6916ee98a)


