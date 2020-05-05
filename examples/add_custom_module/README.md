# Quickstart

Change directory to this folder and run:
```bash
nexula -r nexula.yaml
```

The yaml file contains:
```yaml
nexula_data:
  data_choice_type: 'manual_split'
  data_reader_type: 'read_csv'
  data_reader_args:
    train:
      file: 'data/train.csv'
    dev:
      file: 'data/dev.csv'
    test:
      file: 'data/test.csv'
  data_pipeline:
    boomer:
      data_representer_func_list_and_args:
        - process: 'shape_printer'
          params:
            init:
              add_text: 'hello world! '
        - process: 'nexus_tf_idf_representer'

nexula_train:
  models:
    - model: 'nexus_boomer_logistic_regression'
  callbacks:
    - callback: 'model_saver_callback'
      params:
        output_dir: 'output/model/'
    - callback: 'benchmark_reporter_callback'
      params:
        output_dir: 'output/'

```
See:
```yaml
- process: 'shape_printer'
          params:
            init:
              add_text: 'hello world! '
```
## look at `custom_module.preprocess_add_data.py`.
```python
from nexula.nexula_inventory.inventory_base import NexusBaseDataInventory


class ShapePrinter(NexusBaseDataInventory):

    name = 'shape_printer'

    def __init__(self, add_text='print', **kwargs):
        super().__init__(**kwargs)
        self.add_text = add_text

    def get_model(self):
        return None

    def __call__(self, x, y, fit_to_data=True, *args, **kwargs):
        """
        Dont modify anything, just print the shape
        """
        print("{}, x shape = {}".format(self.add_text, x.shape))
        return x, y

```
The class contains 'shape_printer' name that we add in the `nexula.yaml`. Anyway, this process will
print the data shape. That's all.

You should have see the output like these:
```
hello world! , x shape = (2,)
```

