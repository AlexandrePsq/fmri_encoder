import os
import yaml



def check_folder(path):
    """Create adequate folders if necessary.
    Args:
        - path: str
    """
    try:
        if not os.path.isdir(path):
            check_folder(os.path.dirname(path))
            os.mkdir(path)
    except:
        pass

def read_yaml(yaml_path):
    """Open and read safely a yaml file.
    Args:
        - yaml_path: str
    Returns:
        - parameters: dict
    """
    try:
        with open(yaml_path, 'r') as stream:
            parameters = yaml.safe_load(stream)
        return parameters
    except :
        print("Couldn't load yaml file: {}.".format(yaml_path))
    
def save_yaml(data, yaml_path):
    """Open and write safely in a yaml file.
    Args:
        - data: list/dict/str/int/float
        - yaml_path: str
    """
    with open(yaml_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    
def write(path, text, end='\n'):
    """Write in the specified text file.
    Args:
        - path: str
        - text: str
        - end: str
    """
    with open(path, 'a+') as f:
        f.write(text)
        f.write(end)
