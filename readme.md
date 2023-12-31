# BCI GNN Composite Damage Analysis Code
This is the main code made for the completion of my 2023 internship at the Bristol Composites Institute.

Proposed framework: dataset generation and the Graph-LSTM.

<img src="/readme/flowchart.png" width=75% height=75%>


## Input File Generation
- Run `get_odb_data.py` for each .odb and place generated folder in `/data/`.
- Specify the .odb location and wrinkle amplitude in the file.

## Model
The graph LSTM model is instantiated with:
- `GraphLSTM(enc_hidden, enc_out, lstm_hidden, out_features, sequence_length, data, device, num_layers_enc, num_layers_lstm)`

## Training
- The key file is `train.py`.  Network architecture and hyperparameters can be specified with input arguments found with `-h`.
- Running `train.py` with a `-g` flag will generate a temporal dataset `data.dat` file for all folders present in `/data/`.  No flag loads the `data.dat` file.
- When training is completed or after a keyboard interrupt, the model weights are exported as a .pt network weights file. These files can be reloaded for further training using the `-c` flag, followed by the file path.
- All .pt and .dat files are stored in `/export/`

## Testing
- When testing, the `data.dat` file must be present in `/export/`.  
- The model parameters must be consistent with the trained model parameters and the filename of the weights `.pt` file must be specified.

## Results
- Left: Correlation between ground truth and predicted values for matrix and delamination failure indices.
    -  $R^2$ scores of 0.95 and 0.92 respectively
- Right: Error histogram for matrix and delamination failure index predictions
    -  RMS scores of 0.06 and 0.03 respectively

<p float="left">
  <img src="/readme/results1.png" width="39.5%" />
  <img src="/readme/results2.png" width="50%" /> 
</p>
