# C147/247 Final Project Code
### Winter 2025 - _Professor Jonathan Kao_

## Code Organization

Each architecture used has 3 python files modified, at most: transformer.py, modules,py, and lightning.py. Therefore, all 3 are provided in each architecture's folder. We also modified various parts of the config folder per architecture, so a corresponding copy of config is also within each arhcitecture's folder.

In addition, there are some files that we permanently changed for the project, but we did not change them for any of our architectures our experiments. One example is data.py (changed to handle our to ToTensor function). We included all the other code in the project in the folder "environment," which is complete except for the actually having data within the data folder, the 3 python files listed prior, and the config file, which are found in each architecture's folders.

## Modification for experiements

Running the objective 1 experiments were straightforward. The code used to plot is called plot_errors, but this is done after we extracted CSV data from the Tensorboard. We still include the code for plotting.

For objective 2, we simply deleted the last 8 or 12 training files, depending on if we were investigating 8 or 4 sessions.

For altering thenumber of channels, there are three places where change was made, both to the baseline code (not provided, because it is identical to the project prompt's github code) and to LSTM_objective2:
 - num_features in config/model/tds_conv_ctc.yaml divides by how many the same factor as number of channels
 - the hard-coded NUM_CHANNELS value in the class the model is in changes (the model is in lightning.py).
 - ToTensor's channel_stride changes to 1 for all channels, 2 for 8 channels, 4 for 4 channels, 8 for 2 channels, and 16 for 1 channel.

 For altering the training data to use a second user, copy the text from config/user/pseudo_zeroshot.yaml into the single_user.yaml file in the same folder. Note that the actual data isn't included in this github repo.
