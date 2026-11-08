"""Train-source molecular graph diagnostics; no biological validation implied."""
from rdkit import Chem


def assess_molecules(generation, targets):
    """Count termination separately from nonempty validity and isomeric recovery.

    Recovery denominator includes every condition. Uniqueness denominator includes
    only valid outputs. Targets are accessed only after generation by the caller.
    """
    smiles = tuple(generation['smiles'])
    terminated = tuple(generation['terminated'])
    targets = tuple(targets)
    if not targets or len(smiles) != len(targets) or len(terminated) != len(targets):
        raise ValueError('nonempty aligned generation and targets required')

    def graph(text):
        if not isinstance(text, str) or not text.strip():
            return None
        mol = Chem.MolFromSmiles(text)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

    expected = [graph(t) for t in targets]
    if any(t is None for t in expected):
        raise ValueError('valid nonempty source targets required')
    actual = []
    for text, done in zip(smiles, terminated):
        if type(done) is not bool or (done and not isinstance(text, str)) or (not done and text is not None):
            raise ValueError('termination and decoded output disagree')
        actual.append(graph(text) if done else None)
    valid = [x for x in actual if x is not None]
    recovered = sum(x is not None and x == y for x, y in zip(actual, expected))
    return dict(condition_count=len(targets), terminated_count=sum(terminated),
                valid_nonempty_count=len(valid), unique_valid_isomeric_graphs=len(set(valid)),
                exact_isomeric_graph_recovery_count=recovered,
                recovery_fraction_all_conditions=recovered / len(targets),
                valid_unique_fraction=len(set(valid)) / len(valid) if valid else None)


def predict_molecules(model, views):
    """Target-free greedy generation and within-modality condition rotation.

    BOS logits measure condition sensitivity without teacher forcing. Rotation is
    a descriptive intervention on the supplied condition batch, not evidence of
    correct biological control. Singleton modalities cannot test sensitivity.
    Restore heterogeneous training modes and CPU/initialized CUDA RNG on errors.
    """
    import torch
    from .profile_decoder import ProfileSelfiesDecoder
    if not isinstance(model, ProfileSelfiesDecoder):
        raise ValueError('native profile SELFIES decoder required')
    views=tuple(views)
    groups={}
    for i,v in enumerate(views):groups.setdefault(v.modality,[]).append(i)
    rotation=list(range(len(views)));eligible=[]
    for indices in groups.values():
        if len(indices)>1:
            eligible.extend(indices)
            for i,j in zip(indices,indices[1:]+indices[:1]):rotation[i]=j
    modes=tuple((m,m.training) for m in model.modules())
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices),torch.no_grad():
            model.eval()
            views=model._views(views)
            rotated=tuple(views[i] for i in rotation)
            p=next(model.parameters())
            prefix=torch.full((len(views),1),model.codec.bos,device=p.device,dtype=torch.long)
            logits=model(views,prefix)[:,0].detach().cpu().clone()
            control=model(rotated,prefix)[:,0].detach().cpu().clone()
            generation=model.generate(views);other=model.generate(rotated)
            delta=(logits[eligible].double()-control[eligible].double()).abs()
            return dict(generation=generation,rotated_generation=other,
                bos_logits=logits,rotated_bos_logits=control,rotation_indices=tuple(rotation),
                sensitivity=dict(eligible_condition_count=len(eligible),
                    singleton_condition_count=len(views)-len(eligible),
                    mean_abs_bos_logit_change=float(delta.mean()) if eligible else None,
                    max_abs_bos_logit_change=float(delta.max()) if eligible else None,
                    changed_token_sequence_count=sum(generation['token_ids'][i]!=other['token_ids'][i] for i in eligible)))
    finally:
        for m,training in modes:m.training=training
