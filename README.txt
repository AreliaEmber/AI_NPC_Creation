The first step to be able to create a new NPC with this system is to install all of the necessary programs, and download the code from GitHub. The installs needed are as follows:

(1) Unreal Engine10 version 5.3.2 (https://www.unrealengine.com/de)
(2) A python environment of version 3.7.9 according to the specifications in the readme under text-to-motion, which is copied from the readme file here: https://github.com/EricGuo5513/text-to-motion. It’s recommended to setup this environment using anaconda, as that was used in the implementation of the prototype and worked well, but it isn’t necessary
(3) Add the following libraries to the python environment after it is fully set up: (i) pyttsx3, (ii) openai.

Once all the relevant programs are installed, an OpenAI API key needs to be acquired (see https://platform.openai.com/api-keys) and ideally saved as an environment variable to your system so that it can be accessed automatically when running the program. 

To do this on windows, go to the control panel in system/advanced system settings/advanced/environment variables, select system variables, then new, then enter the name of the variable as "OPENAI_API_KEY" and enter the API key as the value.

Once this is all set up, it should be possible to execute the python side of the program by opening the python environment that was setup to the directory "text-to-motion" in the files from the GitHub download, and executing this line in the console: "python gen_motion_script.py –name Comp_v6_KLD01 –text_file input.txt –repeat_time 1 –ext customized –gpu_id 0".
It should be possible to start the Unreal Engine at this point as well, but the project files may need to be regenerated first. That can be achieved by navigating to the Unreal Engine project files (MyProject3 for the AI NPC) and right clicking on the .uproject file, then selecting "Generate Visual Studio project files" from the drop down that appears. This option will only be available if Unreal Engine is installed.

Once all of the above starts without errors, creating a new NPC can be done by opening the "initial_messages.txt" file from the folder "text-to-motion" and changing the first two lines of that file to the initial messages that the NPC should receive. Ideally both messages should describe the scenario that the NPC finds itself in, and any important information the NPC should be aware of. Once that file has been edited and saved, the NPC can be tested by running the python file as described above, and opening and running the Unreal Engine project. It’s possible to communicate with the NPC via the console of the python file, and the movements generated alongside the audio response can be seen in the Unreal Engine project.
