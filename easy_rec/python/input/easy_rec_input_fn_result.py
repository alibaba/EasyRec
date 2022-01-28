class EasyRecInputFnResult:
    def __init__(self, result, collections):
        self._result = result
        self._collections = collections
    
    @property
    def result(self):
        return self._result

    @property
    def collections(self):
        return self._collections
