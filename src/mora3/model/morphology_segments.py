"""Immutable bounded training segments for trusted local morphology experiments.

The receipt is written last and flushed to disk. Partial directories are retained
for inspection but never admitted for resume. No existing artifact is replaced.
This is a checksum/integrity contract, not authentication of untrusted pickles.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import torch

SCHEMA='mora3.morphology-segment.v1'


def train_segment(trainer,steps,output):
    """Advance exactly ``steps`` updates and persist complete continuation state."""
    if type(steps) is not int or steps<=0:
        raise ValueError('Positive integer segment horizon required')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    start_step=trainer.steps
    start=time.perf_counter()
    losses=[trainer.step() for _ in range(steps)]
    if trainer.config['device'].startswith('cuda'):
        torch.cuda.synchronize(trainer.config['device'])
    training_seconds=time.perf_counter()-start
    payload=dict(schema=SCHEMA,start_step=start_step,end_step=trainer.steps,
                 losses=losses,training_seconds=training_seconds,state=trainer.snapshot())
    path=output/'state.pt'
    with path.open('xb') as stream:
        torch.save(payload,stream)
        stream.flush(); os.fsync(stream.fileno())
    receipt=dict(schema=SCHEMA,start_step=start_step,end_step=trainer.steps,
                 sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
    with (output/'receipt.json').open('x') as stream:
        json.dump(receipt,stream,indent=2); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    descriptor=os.open(output,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
    return receipt


def restore_segment(trainer,output):
    """Restore a complete hash-checked, locally trusted segment with matching data."""
    output=Path(output)
    try: receipt=json.loads((output/'receipt.json').read_text())
    except (OSError,ValueError) as error: raise ValueError('Missing/invalid segment receipt') from error
    raw=(output/'state.pt').read_bytes()
    if receipt.get('schema')!=SCHEMA or receipt.get('bytes')!=len(raw) or receipt.get('sha256')!=hashlib.sha256(raw).hexdigest():
        raise ValueError('Segment checksum/schema mismatch')
    # Load the exact bytes checked above, avoiding a second file read.
    import io
    payload=torch.load(io.BytesIO(raw),weights_only=False)
    start,end=payload['start_step'],payload['end_step']
    if (payload.get('schema')!=SCHEMA or type(start) is not int or type(end) is not int
        or start<0 or end<=start or len(payload['losses'])!=end-start
        or payload['state']['steps']!=end
        or receipt.get('start_step')!=start or receipt.get('end_step')!=end):
        raise ValueError('Invalid segment step/history contract')
    trainer.restore(payload['state'])
    return {key:value for key,value in payload.items() if key!='state'}
