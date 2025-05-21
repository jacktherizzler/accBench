# ml accelerator benchmark tool

a minimal benchmarking suite designed to evaluate the performance of machine learning accelerators (cpu, gpu, tpu) across popular deep learning workloads.

## features

- supports benchmarking on pytorch and tensorflow
- evaluates training and inference times across cnn, rnn, and transformer models
- includes synthetic and real datasets for controlled benchmarking
- supports csv-based logging and visualizations of latency and throughput
- cross-platform support for cpu, nvidia gpu, and tpu (colab integration)

## setup

```bash
git clone https://github.com/yourusername/ml-accelerator-benchmark-tool
cd ml-accelerator-benchmark-tool
pip install -r requirements.txt
```

## usage

```bash
python benchmark.py --backend pytorch --model resnet --device gpu
```

## benchmarks included

	• cnn: resnet18, mobilenetv2
	• rnn: lstm, gru
	• transformer: vit, bert
