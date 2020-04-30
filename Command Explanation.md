# Command Explaination
This markdown will explain about the command that must be written if you want to run the yaml file.

# YAML Skeleton
```yaml
nexula_data: # Pointer for controlling the data pipeline
  data_choice_type : <DATA_CHOICE> # how to setup the data. Currently the only possible one is `manual_split`
  data_reader_type : <DATA_READER> # how to read the data. the data must follow
  data_reader_args : <DATA_READER_ARGS> # arguments data_reader_type. 
    train: # DATA READER for training set. The following should follow data_reader_type args
    dev: # For development set
    test: # For validation set
  data_pipeline: # Pipeline
    boomer: # Data pipeline for shallow learning
        data_preprocesser_func_list_and_args: # Preprocessing. You can delete this if you don't want to include any preprocesing
          - process: <PREPROCESSER_NAME> # your preprocesser name
            params:
              init: # Arguments passed to the class. Used on instantiating the object
              call: # Arguments passed to the class. Used on calling the object
        data_representer_func_list_and_args: # Representer. You can delete this if you don't want to include any preprocesing
          - process: <REPRESENTER_NAME> # your feature representer name
            params:
              init: # Arguments passed to the class. Used on instantiating the object
              call: # Arguments passed to the class. Used on calling the object
    millenial: # Data pipeline for deep learning
        data_preprocesser_func_list_and_args: # Preprocessing. You can delete this if you don't want to include any preprocesing
          - process: <PREPROCESSER_NAME> # your preprocesser name
            params:
              init: # Arguments passed to the class. Used on instantiating the object
              call: # Arguments passed to the class. Used on calling the object
        data_representer_func_list_and_args: # Representer. You can delete this if you don't want to include any preprocesing
          - process: <REPRESENTER_NAME> # your feature representer name
            params:
              init: # Arguments passed to the class. Used on instantiating the object
              call: # Arguments passed to the class. Used on calling the object
nexula_train: # Pointer on controlling the training process
    models: 
      - model: <MODEL_NAME>
        params:
          init: # used when instantiating the model object
          call: # used when fitting the model.
    callbacks:
      - callback : <CALLBACK_NAME>
        params: # callback parameter, no init and call in callbacks unlike others
    lightning_callbacks:
      - callback: <LIGHTNING_CALLBACK> # All callback from pytorch lightning callbacks
        params: # Callback parameter
```

## Note
* Lightning_callbacks will be called on deep learning process
* 'call' Models args in deep learning nexula modules is the arguments for Trainer.fit() in Pytorch Lightning

# What are the possible preprocessing, model etc?
See `nexula.nexula_inventory.inventory_translator` to see all possible input.

# Complex Examples
```yaml
nexus_data:
  data_choice_type: 'manual_split'
  data_reader_type: 'read_csv'
  data_reader_args:
    train:
      file: 'tests/dummy_data/train.csv'
    dev:
      file: 'tests/dummy_data/dev.csv'
    test:
      file: 'tests/dummy_data/test.csv'
  data_pipeline:
    boomer:
      data_preprocesser_func_list_and_args:
        - process: 'nexus_basic_preprocesser'
          params:
            init:
              operation:
                - 'lowercase'
      data_representer_func_list_and_args:
        - process: 'nexus_tf_idf_representer'
    millenial:
      data_preprocesser_func_list_and_args:
        - process: 'nexus_basic_preprocesser'
          params:
            init:
              operation:
                - 'lowercase'
      data_representer_func_list_and_args:
        - process: 'nexus_millenial_representer'
          params:
            init:
              tokenizer_name: 'nltk_wordtokenize'
              seq_len: 50
              lowercase: true
              batch_size: 1
              vocab_size: 10000
              min_freq: 1
              fit_first: 'fit_all_dataset'

nexus_train:
  models:
    - model: 'nexus_boomer_logistic_regression'
    - model: 'nexus_boomer_linear_svc'
    - model: 'nexus_millenial_ccn1d_classification'
      params:
        init:
          embedding_dim: 100
          n_filters: 100
          filter_sizes:
            - 3
            - 4
            - 5
          dropout: 0.2
        call:
          gpus:
            - 0
          weight_summary: 'full'
          max_epochs: 20
  callbacks:
    - callback: 'model_saver_callback'
      params:
        output_dir: 'output'
    - callback: 'benchmark_reporter_callback'
      params:
        output_dir: 'output'
  lightning_callbacks:
    - callback: 'EarlyStopping'
      params:
        monitor: 'val_loss'
        min_delta: 0.00
        patience: 3
        mode: 'min'

```