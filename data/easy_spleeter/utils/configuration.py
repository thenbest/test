#!/usr/bin/env python
# coding: utf8

""" Module that provides configuration loading function. """

import json

from os import path as osp

from .. import SpleeterError
from ..resources import RESOURCES_ROOT


_EMBEDDED_CONFIGURATION_PREFIX = 'spleeter:'


def load_configuration(descriptor):
    """ Load configuration from the given descriptor. Could be
    either a `spleeter:` prefixed embedded configuration name
    or a file system path to read configuration from.

    :param descriptor: Configuration descriptor to use for lookup.
    :returns: Loaded description as dict.
    :raise ValueError: If required embedded configuration does not exists.
    :raise SpleeterError: If required configuration file does not exists.
    """
    # Embedded configuration reading.
    # if descriptor.startswith(_EMBEDDED_CONFIGURATION_PREFIX):
    #     name = descriptor[len(_EMBEDDED_CONFIGURATION_PREFIX):]
    #     if not loader.is_resource(resources, f'{name}.json'):
    #         raise SpleeterError(f'No embedded configuration {name} found')
    #     with loader.open_text(resources, f'{name}.json') as stream:
    #         return json.load(stream)
    # Standard file reading.   
    
    
    descriptor = osp.join(RESOURCES_ROOT, f"{descriptor}.json")
    if not osp.exists(descriptor):
        raise SpleeterError(f'Configuration file {descriptor} not found')
    with open(descriptor, 'r') as stream:
        return json.load(stream)
