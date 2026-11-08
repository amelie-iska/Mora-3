"""Downstream checks both phase allocations without changing frozen census code."""
from data_intake_full_pubchem_guards import BuildGuard as CensusGuard

class BuildGuard(CensusGuard):
    def check(self,phase,*,reserve=0):
        usage=super().check(phase,reserve=reserve)
        if usage['census_bytes']>self.census_cap:raise RuntimeError('completed census phase cap exceeded; no borrowing')
        if usage['materialized_bytes']>self.total_cap-self.census_cap:raise RuntimeError('materialized phase cap exceeded; no borrowing')
        return usage
