
import onnxruntime as ort
import yaml
import sys
import os

def load_yaml_options(provider, options_folder = "./options", key = "defaults"):
    """Load provider options from YAML if available."""
    options_path = os.path.join(options_folder, f"{provider}.yaml")
    
    if os.path.exists(options_path):
        with open(options_path, "r") as file:
            return yaml.safe_load(file)[key]
    return {}  # Return an empty dictionary if no file exists


def load_model(onnx_model_path, provider='CPU', provider_options=None):
    """
    Load an ONNX model with the specified execution provider and options.
    
    :param onnx_model_path: Path to the ONNX model file.
    :param provider: Execution provider to use ('CPU', 'CUDA', 'TensorRT', etc.).
    :param provider_options: Dictionary of provider-specific options.
    :return: ONNX Runtime InferenceSession
    """
    available_providers = ort.get_available_providers()
    print(f"Available execution providers: {available_providers}")
    
    provider_map = {
        'CPU': 'CPUExecutionProvider',
        'CUDA': 'CUDAExecutionProvider',
        'TensorRT': 'TensorrtExecutionProvider',
    }
    
    if provider not in provider_map:
        print(f"Warning: Provider '{provider}' not recognized.")
        return None
        
    selected_provider = provider_map[provider]
    if selected_provider not in available_providers:
        print(f"Warning: Selected provider '{selected_provider}' is not available.")
        return None

    if provider_options is None:
        provider_options = {}
    
    # Create session with specified provider and options
    print(f"Loading model from path: {onnx_model_path}")
    session = ort.InferenceSession(onnx_model_path, providers=[(selected_provider, provider_options)])
    
    print(f"Model loaded with provider: {selected_provider}")
    return session

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_model.onnx> [provider]")
        sys.exit(1)

    # model_path = '../../../Scrivania/model_database/par/parnet.onnx'

    
    model_path = sys.argv[1]
    provider = sys.argv[2] if len(sys.argv) > 2 else 'CPU'
    
    # provider_options = {
    #     'CUDA': {"cudnn_conv_algo_search": "EXHAUSTIVE", "gpu_mem_limit": 2 * 1024 * 1024 * 1024},
    #     'TensorRT': {"trt_max_workspace_size": 2 * 1024 * 1024 * 1024},
    #     'CPU': {}
    # }
    
    # Load YAML of the provider if any
    options = load_yaml_options(provider, key="timing_cache")
    print(options)

    session = load_model(model_path, provider, options)
    
    # get the name of the first input of the model
    input_name = session.get_inputs()[0].name  
    output_name = session.get_outputs()[0].name

    print('Input Name:', input_name)
    print("Output Name:", output_name)
