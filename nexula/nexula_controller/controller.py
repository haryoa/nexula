import argparse
import logging
from pprint import pformat
from typing import Dict, Tuple, Union, List

from nexula.nexula_data.data_controller import DataController
from nexula.nexula_inventory.inventory_translator import NEXUS_INVENTORIES
from nexula.nexula_inventory.inventory_translator import update_dict_with_custom_module
from nexula.nexula_train.train_controller import TrainerController
from nexula.nexula_utility.utility_import_var import import_class
from nexula.nexula_utility.utility_singleton import CustomNexusTranslatorSingleton
from nexula.nexula_utility.utility_yaml_parser import load_yaml

"""
Choice on how to train data. Currently nexula have:
1. manual_split
    The user prepare train data, dev data and test data by himself
    
"""
DATA_CHOICE_TYPE = {
    0: 'manual_split'
}
logger = logging.getLogger('nexula')


class NexusController:

    def main(self):
        """
        Main program of the nexula.
        Currently only supports running yaml program
        """
        parser = self.get_argument_parser()
        args = parser.parse_args()
        self.create_logger_object(args.verbose)
        logger.info("Input {}".format(args))
        self.customize_custom_module(args.custom_module)
        self._main_helper(args)

    def customize_custom_module(self, custom_module: str):
        """
        Add user custom module into nexula_inventory translator.
        With this action, we can use the user custom module.
        Note that the custom module must inherit one of these classes:
        - NexusBaseDataInventory
        - NexusBaseModelInventory
        - NexusBaseCallbackInventory

        Parameters
        ----------
        custom_module: str
            The path to the custom module. e.g: 'x/b/c'. We recommend to place the custom module

        Returns
        -------

        """
        if custom_module is not None:
            # Validate if path is directory
            from pathlib import Path
            if not Path(custom_module).is_dir():
                raise FileNotFoundError("{} directory is not found".format(custom_module))
            all_modules_dict = CustomNexusTranslatorSingleton.recursive_get_all_module(custom_module)
            if len(all_modules_dict) == 0:
                logger.warning("Your custom module does not have anything that inherit nexula base class.")
            test = CustomNexusTranslatorSingleton(all_modules_dict)
            update_dict_with_custom_module(test.custom_translation)

    def create_logger_object(self, verbose):
        from nexula.nexula_logging.nexus_log import NexusLogger
        level_log = 'debug' if verbose else 'info'
        nl = NexusLogger()
        nl.create_logger('nexula', low_level=level_log, colored=True)

    def test_main(self, args: list):
        """
        Used for testing utility
        Parameters
        ----------
        args : list
            List argument to be passed to main program.
            i.e : ['-r', 'test.yaml']
        """
        parser = self.get_argument_parser()
        args = parser.parse_args(args)
        logger.info("=User input is= \n{}\n".format(pformat(args)))
        self._main_helper(args)

    def _main_helper(self, args: argparse.Namespace):
        """
        Parse yaml input and run the command

        Parameters
        ----------
        args: argparse.Namespace
            Arguments from user input
        """
        yaml_file = args.run_yaml
        command_dict = load_yaml(yaml_file)
        NexusCommander().run_command(command_dict)

    @staticmethod
    def get_argument_parser() -> argparse.ArgumentParser:
        """
        Parser goes here

        Returns
        =======
        argparse.ArgumentParser
            Parser of the program input
        """
        parser = argparse.ArgumentParser(prog='nexula',
                                         usage='python -m %(prog)s [options]',
                                         description='Nexus Main Program')
        parser.add_argument('-r',
                            '--run-yaml',
                            help='Yaml file as a command of the nexula', required=True)
        parser.add_argument('-v',
                            '--verbose',
                            help='Add verbosity (debug to logger)', required=False, action='store_true')
        parser.add_argument('-c',
                            '--custom-module',
                            help='Add custom module', required=False)

        return parser


class NexusCommander:
    """
    Run the command in the dictionary. The dictionary shall follow the rule in this URL.
    TODO : Give markdown URL which explain the rule
    """

    FUNC_DATACONTROLLER_NAME = ['data_preprocesser_func_list_and_args', 'data_representer_func_list_and_args',
                                'data_postprocesser_func_list_and_args']

    def run_command(self, dictionary: Dict[str, object]):
        """
        Main commander of Nexus.
        Run all operation defined in the `dictionary` which is defaulted by extracting the yaml file.
        The dictionary should follow these rule

        Parameters
        ----------
        dictionary : Dict[object]
            dictionary that should follow above format WHICH all of the process and model name will be translated
            into actual function/class according to `inventory_translator`
        """
        # Separate millenial and boomer data_pipeline. Later, need to be elegant way
        model_runtypes = list(dictionary['nexula_data']['data_pipeline'].keys())
        list_run_dicts = []
        for run_type in model_runtypes:
            dict_runtype = dict(nexula_data=dict())
            self._copy_data_reader(dict_runtype, dictionary, run_type)
            self._construct_dict_runtime(dict_runtype, dictionary, run_type)
            list_run_dicts.append(dict_runtype)
        for run_dict in list_run_dicts:
            from pprint import pformat
            logger.info("Running \n {} \n".format(pformat(run_dict)))
            input_data_dict = run_dict['nexula_data']
            data_choice_args = self._run_nexus_data_controller(input_data_dict)
            logger.info("Running : \n {} \n".format(pformat(run_dict['nexula_train'])))
            self._run_nexus_train_controller(run_dict['nexula_train'], data_choice_args,
                                             run_dict['nexula_data']['data_choice_type'])

    def _copy_data_reader(self, input_dict: dict, input_ref: dict, dataloader_key: str = 'boomer'):
        """
        Separate boomer and millenial method into 2 dicts.
        This function will copy some of value in original dict into these dictionary.
        """
        input_dict['nexula_data']['data_choice_type'] = input_ref['nexula_data']['data_choice_type']
        input_dict['nexula_data']['data_reader_type'] = input_ref['nexula_data']['data_reader_type']
        input_dict['nexula_data']['data_reader_args'] = input_ref['nexula_data']['data_reader_args']
        for key, value in input_ref['nexula_data']['data_pipeline'][dataloader_key].items():
            input_dict['nexula_data'][key] = value

    def _construct_dict_runtime(self, input_runtype_dict: dict, input_dictionary: dict, type: str) -> dict:
        """
        Separate input on 'nexula_train' according to the model type.

        Parameters
        ----------
        input_runtype_dict:
            Runtype dictionary for a model (currently only millenial and boomer)
        input_dictionary L
            Original dictionary
        type :
            model type

        Returns
        -------
        dict
            The constructed dictionary
        """
        input_runtype_dict.setdefault('nexula_train', {'models': []})

        for model in input_dictionary['nexula_train']['models']:
            model_name = model['model']
            model_obj = import_class(*NEXUS_INVENTORIES[model_name])
            model_type = model_obj.type
            if type == model_type:
                input_runtype_dict['nexula_train']['models'].append(model)

        input_runtype_dict['nexula_train'].setdefault('callbacks', input_dictionary['nexula_train']['callbacks'])
        if type == 'millenial':
            input_runtype_dict['nexula_train'].setdefault('lightning_callbacks',
                                                         input_dictionary['nexula_train']['lightning_callbacks'])
        return input_runtype_dict

    def _nexus_data_transformer(self, input_data: Dict[str, object], key: str,
                                dict_translator: Dict[str, Tuple[str, str]]):
        """
        Preprocess the dictionary that will be used in operating Data Controller.
        Add paramters

        Parameters
        ----------
        input_data: Dict[str, object]
            The dictionary input that will be manipulated
        key: str
            The key in dictionary that want to be preprocessed
        dict_translator: Dict[str, Tuple[str,str]]
            The translator from inventory_translator, which translate the process (str) into class

        """
        if key not in input_data:
            input_data[key] = []
        for i, proc_data in enumerate(input_data[key]):
            # ADD EXCEPTION CLASS NOT FOUND
            input_data[key][i]['process'] = import_class(*dict_translator[
                input_data[key][i]['process']
            ])
            input_data[key][i].setdefault('params', {})
            input_data[key][i]['params'].setdefault('init', {})
            input_data[key][i]['params'].setdefault('call', {})

    def _run_nexus_data_controller(self, input_data_dict:Dict[str, object]) -> Dict[str, object]:
        """
        RUN data controller

        Parameters
        ----------
        input_data_dict: Dict[str, object])
            `nexula_data` value from the main dictionary. The dictionary must contains
            all possible data controller arguments

        Returns
        -------
        Dict[object]
            Arguments that follows `data_choice_type`.
            if data_choice_type is `manual_split` then it will contains train, dev, test dataloader
        """
        # call data_controller.DataController
        key = 'data_preprocesser_func_list_and_args'
        self._nexus_data_transformer(input_data_dict, key, NEXUS_INVENTORIES)

        key = 'data_representer_func_list_and_args'
        self._nexus_data_transformer(input_data_dict, key, NEXUS_INVENTORIES)

        key = 'data_postprocesser_proc'
        self._nexus_data_transformer(input_data_dict, key, NEXUS_INVENTORIES)

        self._add_fit_first_arguments(input_data_dict)
        logger.info("Constructing data loader by using DataController with `manual_split`")
        dc = DataController(input_data_dict['data_choice_type'])

        if dc.data_choice_type == DATA_CHOICE_TYPE[0]:
            train_dataloader, validation_dataloader, test_dataloader = dc.construct_data_loader(**input_data_dict)
            return {
                'train_dataloader': train_dataloader,
                'validation_dataloader': validation_dataloader,
                'test_dataloader': test_dataloader
            }
        else:
            return dict()

    def _fix_dict_train_param(self, dict_param: Dict[str, object]):
        """
        Fix key in `nexula_train` dictionary if not present

        Parameters
        ----------
        dict_param: Dict[str, object]
            The `nexula_train` dictionary that want to be fixed

        """
        if 'init' not in dict_param:
            dict_param['init'] = {}
        if 'call' not in dict_param:
            dict_param['call'] = {}

    def _get_new_train_dict(self) -> Dict[str, Union[list, Dict[str, object]]]:
        """
        Returns
        -------
        Dict[str, object]
            New dictionary for `nexula_train`
        """
        return {
            'models' : [],
            'models_args' : [],
            'data_choice_args' : dict(),
            'callbacks' : []
        }

    def _construct_callback(self, input_model_dict: Dict[str, object],
                            constructed_dict: Dict[str, Union[list, Dict[str, object]]]):
        """
        Instantiate object of callback in dictionary, since the TrainerController needs it to be instantiated.

        Parameters
        ----------
        input_model_dict: : Dict[str, object]
            `nexula_data` original dictionary
        constructed_dict: Dict[str, Union[list, Dict[str, object]]]
            Constructed dictionary which satisfies the arguments of TrainerController

        Returns
        -------

        """

        for callback_dict in input_model_dict['callbacks']:
            class_cb = import_class(*NEXUS_INVENTORIES[callback_dict['callback']])
            constructed_dict['callbacks'].append(class_cb(**callback_dict['params']))

    def _fix_dict_train_model(self, input_train: Dict[str, Union[list,object]]) -> Dict[str, Union[list, Dict[str, object]]]:
        """
        Fix the `nexula_train` dictionary that satisfies the TrainerController arguments
        Add 'init' and 'call' if not present
        Parameters
        ----------
        input_train: Dict[str, object]
            The `nexula_train` dictionary

        Returns
        -------
        Dict[str, Union[list, Dict[str, object]]]
            The dictionary which satisfies the TrainerController arguments
        """
        constructed_dict = self._get_new_train_dict()
        for model in input_train['models']:
            constructed_dict['models'].append(import_class(*NEXUS_INVENTORIES[model['model']]))
            param = model['params'] if 'params' in model else {'init': {}, 'call': {}}
            self._fix_dict_train_param(param)
            constructed_dict['models_args'].append(param)
        return constructed_dict

    def _run_nexus_train_controller(self, input_model_dict: dict, data_choice_args: dict, data_choice_type: str):
        """
        Run the **Trainer Controller** which will run all of defined model with
        the data

        Parameters
        ----------
        input_model_dict: dict
            The input dictionary (nexula_train in original dictionary)
        data_choice_args: dict
            The argument how to read the data
        data_choice_type: str
            How to read the file. currently, we only supports 'read_csv'
        """
        constructed_dict = self._fix_dict_train_model(input_model_dict)
        constructed_dict['data_choice_type'] = data_choice_type
        constructed_dict['data_choice_args'] = data_choice_args
        self._construct_callback(input_model_dict, constructed_dict)
        logger.info("Running Trainer Controller")
        tc = TrainerController(**constructed_dict)
        tc.run()

    def _add_fit_first_arguments(self, data_dict: Dict[str, Union[str, Dict[str, List[str]]]]):
        """
        Add helper arguments for fit_first arguments that will be needed on fitting the model.
        'fit_first' will need the data reader which will be used to read the file.

        data_dict : nexula_data
        fit_first_args arguments -> {
            'data_choice_type' : 'manual_split,
            'data_reader_args' : {},
            'data_reader_type' :
        }
        """
        for key, processes in data_dict.items():
            if key in self.FUNC_DATACONTROLLER_NAME:
                for process in processes:
                    if 'fit_first' in process['params']['init']:
                        process['params']['init'].setdefault('fit_first_args', {})
                        dict_chosen = process['params']['init']['fit_first_args']
                        dict_chosen['data_reader_type'] = data_dict['data_reader_type']
                        dict_chosen['data_reader_args'] = data_dict['data_reader_args']
                        dict_chosen['data_choice_type'] = data_dict['data_choice_type']
