import streamlit as st
import matplotlib.pyplot as plt
from benchmark import BenchmarkTool

# Initialize benchmark tool
benchmark_tool = BenchmarkTool()

# Model selection
st.sidebar.header('Model Configuration')
model_options = ['ResNet18', 'VGG16', 'AlexNet', 'MobileNetV2']
selected_model = st.sidebar.selectbox('Select model', model_options)

# Load pre-trained models
if selected_model == 'ResNet18':
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
elif selected_model == 'VGG16':
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
elif selected_model == 'AlexNet':
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
elif selected_model == 'MobileNetV2':
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

benchmark_tool.set_model(model)
st.sidebar.success(f'{selected_model} loaded successfully!')

# Dataset selection
st.sidebar.header('Dataset Configuration')
dataset_options = ['CIFAR10', 'MNIST']
selected_dataset = st.sidebar.selectbox('Select dataset', dataset_options)

try:
    if selected_dataset == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    elif selected_dataset == 'MNIST':
        dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    
    benchmark_tool.set_dataset(dataset)
    st.sidebar.success(f'{selected_dataset} loaded successfully!')
except Exception as e:
    st.sidebar.error(f'Error loading dataset: {str(e)}')

# Streamlit app
st.title('ML Accelerator Benchmark Dashboard')

# Device selection
st.sidebar.header('Configuration')
device = st.sidebar.radio('Select device', ['CPU', 'GPU', 'Colab'])

# Benchmark execution
if st.sidebar.button('Run Benchmark'):
    with st.spinner('Running benchmark...'):
        if device == 'CPU':
            benchmark_tool.device = torch.device('cpu')
        elif device == 'GPU':
            benchmark_tool.device = torch.device('cuda')
        else:
            # Placeholder for Colab implementation
            pass

        results = benchmark_tool.run_benchmark()

        # Display results
        st.header('Benchmark Results')
        col1, col2 = st.columns(2)
        with col1:
            st.metric('Training Time', f"{results['training_time']:.4f} s")
        with col2:
            st.metric('Inference Time', f"{results['inference_time']:.4f} s")

        # Initialize graph containers
        if 'training_chart' not in st.session_state:
            st.session_state.training_chart = st.line_chart()
        if 'inference_chart' not in st.session_state:
            st.session_state.inference_chart = st.line_chart()

        # Update graphs
        training_data = pd.DataFrame({'Training Time': [results['training_time']]})
        inference_data = pd.DataFrame({'Inference Time': [results['inference_time']]})
        st.session_state.training_chart.add_rows(training_data)
        st.session_state.inference_chart.add_rows(inference_data)

        # Display metrics
        st.header('Performance Metrics')
        col1, col2 = st.columns(2)
        with col1:
            st.metric('Training Time', f"{results['training_time']:.4f} s")
        with col2:
            st.metric('Inference Time', f"{results['inference_time']:.4f} s")