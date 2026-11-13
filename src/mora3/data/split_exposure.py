"""Exact declared-group exposure against a bounded heldout identity index.

A component crossing assigned roles has a cross-role edge. Comparing every
training row with all indexed heldout keys detects those declared edges without
storing all training identities. It does not discover undeclared relatedness,
resolve namespace equivalence, or deduplicate training records.
"""
from copy import deepcopy

_ROLES = ('train', 'validation', 'test')


def _text(value):
    if type(value) is not str or not value.strip() or len(value) > 4096:
        raise ValueError('Bounded explicit source identity required')
    try:
        value.encode('utf8')
    except UnicodeError as error:
        raise ValueError('UTF-8 source identity required') from error
    return value


class HeldoutGroupExposure:
    def __init__(self, *, max_heldout_keys, max_heldout_rows, max_groups_per_row, max_examples):
        values=(max_heldout_keys,max_heldout_rows,max_groups_per_row,max_examples)
        if any(type(v) is not int or v < 1 for v in values):
            raise ValueError('Explicit positive allocation budgets required')
        self.max_keys,self.max_rows,self.max_groups,self.max_examples=values
        self.groups={};self.identities={};self.rows=[0,0,0];self.missing=[0,0,0]
        self.examples=[];self.phase='heldout';self.train_group_exposed=0;self.train_id_exposed=0

    def _row(self, identity, groups, coordinate):
        _text(identity);_text(coordinate)
        if (type(groups) is not tuple or len(groups) > self.max_groups
                or any(type(k) is not str for k in groups) or len(set(groups)) != len(groups)):
            raise ValueError('Complete unique bounded original group tuple required')
        for key in groups:_text(key)

    def _record(self, mapping, key, role, coordinate, kind):
        bit=1 << role
        if key not in mapping:
            mapping[key]=[bit,coordinate]
            return False
        entry=mapping[key];old=entry[0]
        other=bool(old & ~bit)
        if other and not (old & bit) and len(self.examples)<self.max_examples:
            self.examples.append(dict(kind=kind,identity=key,previous_roles=[_ROLES[i] for i in range(3) if old & (1 << i)],
                current_role=_ROLES[role],previous_coordinate=entry[1],current_coordinate=coordinate))
        entry[0] |= bit
        return other

    def add_heldout(self, identity, role, groups, coordinate):
        if self.phase!='heldout' or type(role) is not int or role not in (1,2):
            raise ValueError('Validation/test indexing must precede training scan')
        self._row(identity,groups,coordinate)
        if self.rows[1]+self.rows[2] >= self.max_rows:
            raise ValueError('Heldout row index exceeds declared bound')
        if len(self.groups)+sum(k not in self.groups for k in groups)>self.max_keys:
            raise ValueError('Heldout group index exceeds declared bound')
        self._record(self.identities,identity,role,coordinate,'task-id')
        for key in groups:self._record(self.groups,key,role,coordinate,'literal-group')
        self.rows[role]+=1;self.missing[role]+=not groups

    def freeze_heldout(self):
        if self.phase!='heldout':raise ValueError('Heldout index can only be frozen once')
        self.phase='train'

    def add_train(self, identity, groups, coordinate):
        if self.phase!='train':raise ValueError('Train scan requires a frozen heldout index')
        self._row(identity,groups,coordinate)
        identity_exposed=identity in self.identities
        if identity_exposed:self._record(self.identities,identity,0,coordinate,'task-id')
        group_exposed=False
        for key in groups:
            if key in self.groups:
                group_exposed=True;self._record(self.groups,key,0,coordinate,'literal-group')
        self.train_id_exposed+=identity_exposed;self.train_group_exposed+=group_exposed
        self.rows[0]+=1;self.missing[0]+=not groups

    def finish(self):
        if self.phase!='train':raise ValueError('Both ordered audit phases must complete')
        self.phase='finished'
        cross_groups=sum(bool(mask & (mask-1)) for mask,_ in self.groups.values())
        cross_ids=sum(bool(mask & (mask-1)) for mask,_ in self.identities.values())
        return dict(schema='Mora3.DeclaredHeldoutGroupExposure/v1',role_rows=dict(zip(_ROLES,self.rows)),
            heldout_unique_groups=len(self.groups),heldout_unique_task_ids=len(self.identities),
            cross_role_group_keys=cross_groups,cross_role_task_ids=cross_ids,
            train_rows_with_group_exposure=self.train_group_exposed,train_rows_with_task_id_exposure=self.train_id_exposed,
            missing_group_rows=dict(zip(_ROLES,self.missing)),all_rows_have_declared_groups=not any(self.missing),
            declared_cross_role_disjoint=(cross_groups==0 and cross_ids==0),examples=deepcopy(self.examples),
            examples_are_bounded=True,within_train_duplicate_audited=False,
            unknown_biological_dependency_excluded=False,training_authorized=False)
