# RNN From Scratch (NumPy)

A complete, end-to-end implementation of a **Recurrent Neural Network (RNN)** built entirely from **NumPy**.  
This project manually implements:

- RNN cell  
- Forward step  
- Unrolling through time  
- Backpropagation Through Time (BPTT)  
- Cross-entropy loss  
- Parameter updates  
- Full training + evaluation pipeline  

No PyTorch. No TensorFlow.  
Just **math → NumPy → functional RNN**.

---

## Project Structure
```
rnn-from-scratch/
 ├── activations.py (anh, dtanh, swish, dswish)
 ├── rnn_cell.py (forward + backward through one timestep)
 ├── rnn_model.py (unfold sequence through time)
 ├── train_utils.py (BPTT, loss, training loop)
 ├── main.py (train + evaluate model)
 └── README.md
```
