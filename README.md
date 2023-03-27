<strong>MaRVeL:</strong><br />
In this repository, we provide an implementation for the paper "Maximal Robust Neural Network Specifications via Oracle-guided Numerical Optimization" <a href="https://anankabaha.files.wordpress.com/2022/12/maximal_robust_neural_network-specifications_via_oracle_guided_numerical_optimization.pdf">MaRVeL paper</a>. The repository owner is anan.kabaha@campus.technion.ac.il. 

<strong>Clone MaRVeL:</strong><br />
<div style="background-color: #f2f2f2; padding: 1px;">
  <pre style="font-family: 'Courier New', monospace; font-size: 14px;">
  git clone https://github.com/ananmkabaha/MaRVeL.git
  mkdir MaRVeL_system
  mv MaRVeL MaRVeL_system
  cd MaRVeL_system
</pre>
</div>

<strong>Installat ERAN dependencies:</strong><br />
Please follow the instructions in https://github.com/eth-sri/eran <br/>
MaRVeL assumes that ELINA's folder exists into the folder of MaRVeL_system. 

<strong>MaRVeL parameters:</strong><br />
--netname: the network name, the extension can be only .onnx
--dataset: the dataset, can be either mnist, fmnist, cifar10, contagio, or syn
--timeout: the total certification timeout
--max_iterations: the maximal number of optimization steps
--milp_num_of_neurons: the number of milp neurons per layer
--milp_time_limit: the time limit to solve a single milp problem
--lr: the learning rate
--lambda: the tradeoff parameter
--lambda_alpha: the decay rate of the tradeoff parameter
--lambda_min: the minimal value of the tradeoff parameter
--parallelization: the flag to run milp problems in parallel mode 0/1
--output_file: the output file
--samples_num: the number of samples to analyze
--t_size: the minimal difference between two optimization steps


<strong>Examples:</strong><br />
python3 run.py --netname ./models/SYN1.onnx --dataset syn --lr 0.01 <br />
python3 run.py --netname ./models/MNIST1.onnx --dataset mnist --lr 0.1<br />

We note that the best values for the lr paramter are between [0.01-0.2].<br />
For mnist, fmnist, and contagio we recommend the values between [0.1-0.2], and for syn and cifar10 we recommend the values between [0.01-0.05]. 

