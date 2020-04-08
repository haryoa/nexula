from nexus.nexus_train.train_controller import TrainerController
from nexus.nexus_utility.utility_import_var import import_class
from nexus.nexus_inventory.inventory_translator import NEXUS_INVENTORY_PREPROCESSER as nexus_preprocesser
from nexus.nexus_inventory.inventory_translator import NEXUS_INVENTORY_FEATURE_REPRESENTER as nexus_representer
from nexus.nexus_inventory.inventory_translator import NEXUS_INVENTORY_POSTPROCESSER as nexus_postprocesser
from nexus.nexus_data.data_controller import DataController
from nexus.nexus_inventory.inventory_translator import NEXUS_INVENTORY_MODEL as nexus_inventory_model
from nexus.nexus_inventory.inventory_translator import NEXUS_CALLBACKS as nexus_callbacks_dict
from nexus.nexus_utility.utility_yaml_parser import load_yaml

from typing import Dict, Tuple, Union, List
import argparse

# TODO REFACTOR
"""
Choice on how to train data. Currently nexus have:
1. manual_split
    The user prepare train data, dev data and test data by himself
    
"""
DATA_CHOICE_TYPE = {
    0 : 'manual_split'
}


class NexusController:

    def main(self):
        """
        Main program of the nexus.
        Currently only support running yaml program
        """
        parser = self._get_argument_parser()
        args = parser.parse_args()
        self._main_helper(args)

    def test_main(self, args:list):
        """
        Used for testing utility
        Parameters
        ----------
        args : list
            List argument to be passed to main program.
            i.e : ['-r', 'test.yaml']
        """
        parser = self._get_argument_parser()
        args = parser.parse_args(args)
        self._main_helper(args)

    def _main_helper(self, args):
        yaml_file = args.run_yaml
        command_dict = load_yaml(yaml_file)
        NexusCommander().run_command(command_dict)

    def _get_argument_parser(self) -> argparse.ArgumentParser:
        """
        Parser goes here

        Returns
        =======
        argparse.ArgumentParser
            Parser of the program input
        """
        parser = argparse.ArgumentParser(prog='nexus',
                                         usage='python -m %(prog)s [options]',
                                         description='Nexus Main Program')
        parser.add_argument('-r',
                            '--run-yaml',
                            help='Yaml file as a command of the nexus', required=True)

        return parser


class NexusCommander:

    def run_command(self, dictionary: Dict[str, object]):
        """
        Main commander of Nexus.
        Run all operation defined in the `dictionary`. The dictionary must follow the format as follow

        ```
        'nexus_data' :
            data_choice_type : str (How the data to be prepared)
            data_reader_type : str (File reader function, see `nexus.nexus_data.data_reader`. Use the name of the
                                function.
            data_reader_args : The format follows the data_choice_type
            data_preprocesser_func_list_and_args : list<OPTIONAL>
                List of the preprocesser of the data. The dictionary contains
                 'process' : str (process name, see nexus.nexus_inventory.inventory_translator and look the
                    preprocesser)
                  'params' : Dict ( parameters of the class). The Dict is composed of:
                    - init (arguments to be used when the object is created)
                    - call (arguments to be used when the object is called that satisfy __call__ args)
                  )
            data_representer_func_list_and_args : list<OPTIONAL>
                list of feature extractor or representor for the data. The format also follow
                    data_preprocesser_func_list_and_args
            data_postprocesser_proc : list<Optional>
                List of operation that can be done after the data's feature has been represented.
                STILL NOT IMPLEMENTED
        'nexus_train' :
            'models': list (list of models (in dict format)
                        that want to be tried that follow nexus_translator model name). Possible arguments:
                    model : the model name
                    key :
        ```


        Parameters
        ----------
        dictionary : Dict[object]
            dictionary that should follow above format WHICH all of the process and model name will be translated
            into actual function/class according to `inventory_translator`
        """
        # Parse arg param
        input_data_dict = dictionary['nexus_data']
        data_choice_args = self._run_nexus_data_controller(input_data_dict)
        self._run_nexus_train_controller(dictionary['nexus_train'] , data_choice_args,
                                         dictionary['nexus_data']['data_choice_type'])

    def _nexus_data_transformer(self, input_data: Dict[str, object], key: str, dict_translator: Dict[str, Tuple[str,str]]):
        """
        Preprocess the dictionary that will be used in operating Data Controller.
        The format will satisfy the classes arguments. Add required key in dictionary if not present.

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
            if 'params' not in input_data[key]:
                input_data[key][i]['params'] = {}
            if 'init' not in input_data[key][i]:
                input_data[key][i]['params']['init'] = {}
            if 'call' not in input_data[key][i]:
                input_data[key][i]['params']['call'] = {}

    def _run_nexus_data_controller(self, input_data_dict:Dict[str, object]) -> Dict[str, object]:
        """
        RUN data controller

        Parameters
        ----------
        input_data_dict: Dict[str, object])
            `nexus_data` value from the main dictionary. The dictionary must contains
            all possible data controller arguments

        Returns
        -------
        Dict[object]
            Arguments that follows `data_choice_type`.
            if data_choice_type is `manual_split` then it will contains train, dev, test dataloader
        """
        # call data_controller.DataController
        key = 'data_preprocesser_func_list_and_args'
        self._nexus_data_transformer(input_data_dict, key, nexus_preprocesser)

        key = 'data_representer_func_list_and_args'
        self._nexus_data_transformer(input_data_dict, key, nexus_representer)

        key = 'data_postprocesser_proc'
        self._nexus_data_transformer(input_data_dict, key, nexus_postprocesser)

        dc = DataController(input_data_dict['data_choice_type'])

        if dc.data_choice_type == DATA_CHOICE_TYPE[0]:
            train_dataloader, validation_dataloader, test_dataloader = dc.construct_data_loader(**input_data_dict)
            return {
                'train_dataloader' : train_dataloader,
                'validation_dataloader' : validation_dataloader,
                'test_dataloader' : test_dataloader
            }
        else:
            return dict()

    def _fix_dict_train_param(self, dict_param: Dict[str, object]):
        """
        Fix key in `nexus_train` dictionary if not present

        Parameters
        ----------
        dict_param: Dict[str, object]
            The `nexus_train` dictionary that want to be fixed

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
            New dictionary for `nexus_train`
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
            `nexus_data` original dictionary
        constructed_dict: Dict[str, Union[list, Dict[str, object]]]
            Constructed dictionary which satisfies the arguments of TrainerController

        Returns
        -------

        """
        for callback_dict in input_model_dict['callbacks']:
            class_cb = import_class(*nexus_callbacks_dict[callback_dict['callback']])
            constructed_dict['callbacks'].append(class_cb(**callback_dict['params']))

    def _fix_dict_train_model(self, input_train: Dict[str, Union[list,object]]) -> Dict[str, Union[list, Dict[str, object]]]:
        """
        Fix the `nexus_train` dictionary that satisfies the TrainerController arguments

        Parameters
        ----------
        input_train: Dict[str, object]
            The `nexus_train` dictionary

        Returns
        -------
        Dict[str, Union[list, Dict[str, object]]]
            The dictionary which satisfies the TrainerController arguments
        """
        constructed_dict = self._get_new_train_dict()
        for model in input_train['models']:
            constructed_dict['models'].append(import_class(*nexus_inventory_model[model['model']]))
            param = model['params'] if 'params' in model else {'init' : {}, 'call' : {}}
            self._fix_dict_train_param(param)
            constructed_dict['models_args'].append(param)
        return constructed_dict

    def _run_nexus_train_controller(self, input_model_dict, data_choice_args, data_choice_type):
        constructed_dict = self._fix_dict_train_model(input_model_dict)
        constructed_dict['data_choice_type'] = data_choice_type
        constructed_dict['data_choice_args'] = data_choice_args
        self._construct_callback(input_model_dict, constructed_dict)
        tc = TrainerController(**constructed_dict)
        tc.run()

