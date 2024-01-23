# Metadata contents

1) app -> Layout to CSV.exe:

    - IMPORTANT: This app considers that you have images from a **Cytation 5** equipment (specific regex for that). 
    - This folder has an app that is just like the "Layout_to_CSV.ipynb" notebook, just in a easy format to be used in a Windows machine. 
    - Just download and double click and the program will execute:
        - Will ask about the number of wells in your plate;
        - Give a folder containing images as an input;
        - Identifies the channels in that folder (DAPI, GFP, CY5, etc), and asks which nomenclature you want for each channel;
        - It will generate an CSV file with informations from that folder and metadata info.

2) Don't have a Windows machine to execute the app? Use the notebook "Layout_to_CSV.ipynb".