# coding=utf8

import json
import copy
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class Config :
    """
    base class for configuration
    """

    def __init__(self, **kwargs) :
        self.conf = {}
        for key, value in kwargs.items() :
            try :
                if isinstance(value, dict) :
                    setattr(self, key, Config(**value))
                    self.conf[key] = Config(**value)
                else :
                    setattr(self, key, value)
                    self.conf[key] = value
            except AttributeError as error :
                logger.error("Can't set {} with value for {}".format(key, value, self))
                raise error

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "Config" :
        """
        Instantiates a config class from a python disctionary or parameters
        """
        config = cls(**config_dict)
        return config

    @classmethod
    def from_json_file(cls, json_file: str) -> "Config" :
        """
        Instantiates a config class from a json file or parameters
        """
        with open(json_file, "r", encoding="utf-8") as reader :
            text = reader.read()
        config_dict = json.loads(text)
        config = cls(**config_dict)
        return config

    def to_dict(self) -> Dict[str, Any] :
        """
        Serialize this instance to a python dictionary
        """
        output1 = copy.deepcopy(self.__dict__)
        output2 = dict()
        for k, v in output1.items() :
            if isinstance((v, Config)) :
                output2[k] = copy.deepcopy(v.__dict__)
            else :
                output2[k] = v
        return output2

    def to_json_string(self) -> str :
        """
        save this instance to a Json file
        """
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2) + "\n"

    def to_json_file(self, json_file_path: str) :
        with open(json_file_path, "w", encoding="utf-8") as writter :
            writter.write(self.to_json_string())

    def update(self, config_dict: Dict[str, Any]) :
        """
        update attributes
        """
        for key, value in config_dict.items() :
            setattr(self, key, value)

    def __str__(self):
        return str(self.conf)

    def __repr__(self):
        return self.conf


def parse_args(conf) :
    if isinstance(conf, str) :
        config = Config.from_json_file(conf)
    else :
        config = Config.from_dict(conf)
    return config
