# coding=utf8

import json


def parse_conf(f_path):
    return Config(config_file = f_path)


class Config:
    """
    Config load from json file
    """

    def __init__(self, config=None, config_file=None):
        if config_file:
            with open(config_file,'r') as fin:
                config = json.load(fin)

        self.dict = config
        if config:
            self._update(config)

    def __getitem__(self, item):
        return self.dict[item]

    def __contains__(self,item):
        return item in self.dict

    def items(self):
        return self.dict.items()

    def add(self,key,value):
        """
        add key value pair
        """
        self.__dict__[key] = value

    def _update(self,config):
        if not isinstance(config, dict):
            return

        for key in config:
            if isinstance(config[key], dict):
                config[key] = Config(config[key])

            if isinstance((config[key], list)):
                config[key] = [Config(x) if isinstance(x, dict) else x for x in config[key]]

        self.__dict__.update(config)
