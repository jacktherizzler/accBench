import streamlit as st
import matplotlib.pyplot as plt
from benchmark import BenchmarkTool

# Initialize benchmark tool
benchmark_tool = BenchmarkTool()

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