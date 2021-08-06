class MockTensor:
    def __init__(self, value):
        self.value = value

    def __format__(self, format_spec):
        return f"{self.value:{format_spec}}"

    def __str__(self):
        return str(self.value)

    def item(self):
        return self.value

    def backward(self):
        pass
