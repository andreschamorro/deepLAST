import json
import os
from typing import Dict, Union

class Options(dict):
    def __init__(self, *args, **kwargs):
        super(Options, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def get_options_from_json(cls, json_file):
        """
        Get the config from a json file
            :param json_file:
            :return: config(namespace) or config(dictionary)
        """
        # parse the configurations from the config json file provided
        with open(json_file, 'r') as option_file:
            option_dict = json.load(option_file)

        options = cls(option_dict)
        options.summary_dir = os.path.join("run", options.exp_name, "summary/")
        options.checkpoint_dir = os.path.join("run", options.exp_name, "checkpoint/")
        return options

    def todict(self) -> Dict[str, Union[float, int, str]]:
        """Creates a dict from Options.
        Returns:
            Dict[str, Union[float, int, str]]: Dict with all parameters and
                                               values in options
        """
        return self.__dict__.copy()
