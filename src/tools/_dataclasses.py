class DotDict(dict):
    """Custom dictionary class that allows dot notation access for nested structures."""

    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise AttributeError(f"'DotDict' object has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        self[attr] = value