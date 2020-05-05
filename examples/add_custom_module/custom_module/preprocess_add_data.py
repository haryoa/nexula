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
