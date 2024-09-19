import configargparse

def get_config():
    # Adjust the path according to where you place your config folder and what you name your yaml files
    parser = configargparse.ArgParser(default_config_files=['../../config/config.yml'])
    parser.add('-c', '--config', is_config_file=True, help='config file path')
    parser.add('--batch_size', type=int, help='Batch size for data loader', default=64)
    parser.add('--num_images', type=int, help='Number of images to save', default=5)
    parser.add('--output_dir', help='Directory to save sample images', default='./sample_images')
    parser.add('--data_root', help='Root directory for downloading MNIST data', default='./data')
    return parser.parse_args()

