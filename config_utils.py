from ruamel.yaml import YAML

class YParams():
    def __init__(self, config_file, config_profile):
        """
        Inspired by TensorFlow's HParam class

        Create an instance of `YParams` from a YAML config file.

        The parameter type is inferred from the type of the values passed.

        The parameter names are added as attributes of `HParams` object, so they
        can be accessed directly with the dot notation `hparams._name_`.
        """
        with open(config_file) as fp:
            dic_params = YAML().load(fp)[config_profile].items()
            for k, v in YAML().load(fp)[config_profile].items():
                self.add_hparam(k, v)

    def add_hparam(self, name, value):
        """Adds {name, value} pair to hyperparameters.
        Args:
          name: Name of the hyperparameter.
          value: Value of the hyperparameter. Can be one of the following types:
            int, float, string, int list, float list, or string list.
        Raises:
          ValueError: if one of the arguments is invalid.
        """
        # Keys in kwargs are unique, but 'name' could the name of a pre-existing
        # attribute of this object.  In that case we refuse to use it as a
        # hyperparameter name.
        if getattr(self, name, None) is not None:
            raise ValueError(f'Hyperparameter name is already reserved: {name}')
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(f'Multi-valued hyperparameters cannot be empty: {name}')
        setattr(self, name, value)

if __name__ == "__main__":
    yparams = YParams('./configs/test.yml', 'large')
    print(yparams.channels)
    print(yparams.additional_param)

