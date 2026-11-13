"""Build an explicitly synthetic visualization from the REAL inference controller.

No dataset, checkpoint, optimizer or trained quality metric is used. The analytic
fixture score penalizes the first free coordinate; the second retains alternative
latent values. This tests mechanics, not the semantics of a reasoning task.
"""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import torch
from mora3.reasoning.flow_forest_refinement import (
    ForestState,ForestAssessment,DecodedForest,DecodedAssessment,ForestRevision,
    RefinementPolicy,forest_fingerprint,decoded_fingerprint,replace_subtree,
    run_flow_seeded_refinement,
)

C=sha256(b'SYNTHETIC fixture context, not a native biological source').hexdigest()
G=sha256(b'SYNTHETIC two-dimensional diagnostic coordinates').hexdigest()
PIN=sha256(Path(__file__).read_bytes()).hexdigest()


def example(rounds):
    template=ForestState(('condition','left','left-end','right','right-end'),(-1,0,1,0,3),
        ('open','open','terminal','open','terminal'),torch.tensor([[0.,0.]]*5,dtype=torch.float64),
        (0,),('root','computational','computational','computational','computational'),C,G)
    def field(x,t,s):return -.1*x
    def assess(f):
        values=torch.exp(-f.states[:,0].square());score=float(values[1:].mean())
        return ForestAssessment(forest_fingerprint(f),C,PIN,'pass',score,tuple(values.tolist()),PIN,True)
    def edit(f,assessment,*,count,reason,feedback):
        candidates=[i for i in range(len(f.node_ids))if f.parents[i]==0]
        cut=max(candidates,key=lambda i:float(f.states[i,0].square()))
        removed={cut}
        for i,p in enumerate(f.parents):
            if p in removed:removed.add(i)
        indices=sorted(removed);index={p:i for i,p in enumerate(indices)}
        parents=tuple(-1 if i==cut else index[f.parents[i]]for i in indices)
        states=f.states[indices].clone();status=tuple(f.statuses[i]for i in indices)
        proposals=[]
        for j in range(count):
            revised=states.clone();revised[:,0]*=(.25 if j%2==0 else .5)
            child=replace_subtree(f,f.node_ids[cut],revised,parents,status)
            proposals.append(ForestRevision(child,'rule_correction'if reason=='correct'else'subtree_regeneration',f.node_ids[cut]))
        return tuple(proposals)
    def decode(f):
        tokens=tuple(round(float(v)*1000)+100000 for v in f.states.flatten())
        return DecodedForest(tokens,json.dumps(tokens),forest_fingerprint(f),C,PIN)
    def final(f,d):
        values=torch.tensor([(i-100000)/1000 for i in d.tokens],dtype=torch.float64).reshape(-1,2)
        score=float(torch.exp(-values[1:,0].square()).mean())
        return DecodedAssessment(decoded_fingerprint(d),C,PIN,'pass',score,PIN)
    result=run_flow_seeded_refinement(template,field,steps=4,generator=torch.Generator().manual_seed(17),
        policy=RefinementPolicy(6,4,3,rounds,2,22,.35,.35,.8,.1,0),assess=assess,edit=edit,
        decode=decode,assess_decoded=final,field_sha256=PIN,evaluator_sha256=PIN,decoder_sha256=PIN,
        decoded_evaluator_sha256=PIN)
    rows=[]
    for row in result['candidates']:
        f=row['forest'];s=row['assessment']
        rows.append(dict(candidate_id=row['candidate_id'],parent_id=row['parent_id'],round=row['round'],
            operation=row['operation'],score=s.score,status=s.status,node_scores=s.node_scores,
            states=f.states.tolist(),parents=f.parents,node_ids=f.node_ids,statuses=f.statuses,
            clamped=f.clamped,fingerprint=row['candidate_sha256']))
    return dict(rounds=rounds,candidates=rows,output_ids=[d['candidate_id']for d in result['outputs']],
        output_scores=[d['assessment'].score for d in result['outputs']],status=result['status'],
        events=result['events'],candidate_attempts=result['candidate_attempts'],counts=result['inference_counts'],
        velocity_evaluations=result['velocity_evaluations'])


def main():
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1]
    snapshots=[example(rounds)for rounds in(0,1,2)]
    initial=[[x['fingerprint']for x in s['candidates']if x['parent_id']is None]for s in snapshots]
    assert initial[0]==initial[1]==initial[2]
    assert all(len(s['output_ids'])==3 for s in snapshots)
    document=dict(schema='Mora3.SyntheticFlowFoTVisualization/v1',synthetic=True,
        scientific_performance_result=False,training_runs=0,optimizer_updates=0,
        score='mean(exp(-z_first_coordinate_squared)) over every non-condition node; analytic fixture only',
        source_code_sha256=PIN,seed=17,snapshots=snapshots)
    path=root/'tests/project/fixtures/flow-fot-refinement-example.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(document,indent=2)+'\n')
    print(json.dumps(dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,
        snapshots=[dict(rounds=s['rounds'],candidates=len(s['candidates']),outputs=s['output_ids'],scores=s['output_scores'])for s in snapshots])))


if __name__=='__main__':main()
