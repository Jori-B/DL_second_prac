#Nationality identification using deep learning on speech signals
Deep Learning practical 2
By Pim van der Loos, Ruben Spolmink, Jori Blankesteijn, Rik Vegter
April, 2021

This project tries to predict nationalities based on speech signals.

The compressed dataset is provided.
In order to run the code, unzip this dataset and the following experiments can be run (after the dependencies are installed)
1. In order to run the Optuna based method, run python optuna_based.py
2. In order to run the VoxCeleb based method, run python voxceleb_based.py
3. In order to run the AlexNet based method run python AlexNet.py
4.  

gender.py creates weights trained on gender. Pretrained.py loads these weights and uses
these pre-trained weights.
The wav_cut.py file chunks the .wav files in chunks of 5 seconds in order to create data points.
denoise.sh denoises the data based on FT, erases the DC component and removes the silent removal.  
vox1_meta.csv shows meta data about the data.
