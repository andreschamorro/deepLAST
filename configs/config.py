import json
import os

class Options(dict):
    def __init__(self, *args, **kwargs):
        super(Options, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def get_options_from_json(cls, json_file):
    """ Get the config from a json file
            :param json_file:
            :return: config(namespace) or config(dictionary)
        """
        # parse the configurations from the config json file provided
        with open(json_file, 'r') as config_file:
            option_dict = json.load(option_file)

        options = cls(option_dict)
        options.summary_dir = os.path.join("experiments", options.exp_name, "summary/")
        options.checkpoint_dir = os.path.join("experiments", options.exp_name, "checkpoint/")
        return options 
