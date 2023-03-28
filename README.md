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

<strong>Install ERAN's dependencies:</strong><br />
(1) Follow the instructions in https://github.com/eth-sri/eran <br/>
(2) Move ELINA's folder into the folder of MaRVeL_system. 

<strong>MaRVeL parameters:</strong><br />
--netname: the network name, the extension can be only .onnx<br />
--dataset: the dataset, can be either mnist, fmnist, cifar10, contagio, or syn<br />
--timeout: the maximal certification time<br />
--max_iterations: the maximal number of optimization steps<br />
--milp_num_of_neurons: the number of milp neurons per layer<br />
--milp_time_limit: the time limit to solve a single milp problem<br />
--lr: the learning rate<br />
--lambda: the tradeoff parameter<br />
--lambda_alpha: the decay rate of the tradeoff parameter<br />
--lambda_min: the minimal value of the tradeoff parameter<br />
--parallelization: the flag to run milp problems in parallel mode<br />
--output_file: the output file<br />
--samples_num: the number of samples to analyze<br />
--t_size: the minimal difference between two optimization steps<br />


<strong>Examples:</strong><br />
python3 run.py --netname ./models/SYN1.onnx --dataset syn --lr 0.01 --samples_num 50 --lambda 0.99 --milp_time_limit 100 --milp_num_of_neurons 200<br />
python3 run.py --netname ./models/MNIST1.onnx --dataset mnist --lr 0.1--samples_num 50 --lambda 0.99 --milp_time_limit 100 --milp_num_of_neurons 200<br />

We note that the best values for the lr paramter are between [0.01-0.2].<br />
For mnist, fmnist, and contagio we recommend the values between [0.1-0.2], and for syn and cifar10 we recommend the values between [0.01-0.05]. 

