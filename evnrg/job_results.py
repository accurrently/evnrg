from typing import NamedTuple

class JobResults(NamedTuple):

    results: list

    @property
    def failed(self):
        failed = []
        for r in results:
            if r.error:
                failed.append(r)
        return failed
    
    @property
    def succeeded(self):
        good = []
        for r in results:
            if not r.error:
                good.append(r)
        return good