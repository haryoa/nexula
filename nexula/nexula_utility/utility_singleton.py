import glob
import importlib
import logging
import re
from pathlib import Path

from nexula.nexula_inventory.inventory_base import NexusBaseDataInventory, NexusBaseCallbackInventory, \
    NexusBaseModelInventory

logger = logging.getLogger('nexula')


class CustomNexusTranslatorSingleton():
    # TODO REFACTOR
    cls_obj = None
    custom_translation = {}
    base_classes = (NexusBaseDataInventory,
                    NexusBaseCallbackInventory,
                    NexusBaseModelInventory)

    def __init__(self, translation={}):
        print(self.cls_obj)
        if self.cls_obj is None:
            self.custom_translation = translation
            self.cls_obj = self
        else:
            self = self.cls_obj

    @staticmethod
    def recursive_get_all_module(module_dir):
        ctns = CustomNexusTranslatorSingleton
        module_name = module_dir.replace('/', '.')  # refactor, only know '/'
        translation_dict = {}
        globbed = glob.glob("{}/*".format(module_dir))
        for obj in globbed:
            path_obj = Path(obj)
            if path_obj.is_file() and re.match(r'.+\.py$', path_obj.name):
                extracted_name_module = path_obj.name[:-3]
                extracted_name_module = "{}.{}".format(module_name, extracted_name_module)
                translation_instance = ctns.extract_classes(extracted_name_module)
            elif path_obj.is_dir():
                translation_instance = ctns.recursive_get_all_module('{}/{}'.format(module_dir, path_obj.name))
            else:
                translation_instance = {}
            ctns.update_and_validate(translation_dict, translation_instance)

        return translation_dict

    @staticmethod
    def extract_classes(extracted_name_module: str):
        translation_instance = {}
        current_module = importlib.import_module(extracted_name_module)
        for attribute in dir(current_module):
            # Skip if __name, etc in attribute string
            if not re.match(r'__.+?__', attribute):  # base case
                attr = getattr(importlib.import_module(extracted_name_module), attribute)
                if ctns.is_class_in_module(extracted_name_module, attr):
                    translation_instance = ctns.construct_dictionary(attr)
        return translation_instance

    @staticmethod
    def is_class_in_module(module_name, attribute):

        import re
        try:
            class_attr = attribute
            # Get __module__ and check if it is the submodule
            i_module = attribute.__module__  # Not having __module__ means that the attr is not class
            return issubclass(attribute, ctns.base_classes) \
                if re.match(r'^{}$'.format(module_name), i_module) \
                else False  # Handle several cases such as 'a.b.c' to 'zz.vv.a.b.c' \
        except Exception as e:
            return False

    @staticmethod
    def update_and_validate(translation_dict, translation_instance):
        tr_dct_key = translation_dict.keys()
        for ti in translation_instance.keys():
            if ti in tr_dct_key:
                raise Exception("{} has duplicate name on dict {}".format(ti, translation_dict))
        translation_dict.update(translation_instance)

    @staticmethod
    def construct_dictionary(attribute):
        # Make alias_name unique maybe?
        module_name = attribute.__module__
        class_name = attribute.__name__
        alias_name = attribute.name if hasattr(attribute, 'name') else attribute.__name__
        return {alias_name: (module_name, class_name)}


ctns = CustomNexusTranslatorSingleton
