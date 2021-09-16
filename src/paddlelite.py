""" entry point for executing PaddleLite test """
import sys
import json
from model_validator import validate_model

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("must specify a test config file")
        exit(0)
    config_file_path = sys.argv[1];
    with open(config_file_path) as test_config:
        json_object = json.load(test_config)
        for config_file in json_object["config_files"]:
            test_config = {
                "config_file": config_file,
                "engine": "PaddleLite",
                "profile_interation": 100,
                "print_output": True
            };
            validate_model(test_config);
