import importlib.util
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
s=importlib.util.spec_from_file_location('protein_overlay',ROOT/'scripts/data_intake_protein_group_overlay.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_overlay_uses_only_exact_protein_sequence_and_preserves_original_groups():
 old=['accession:P00001','sequence:exact','gene:17'];mapping={'exact':'detected:cluster','17':'must-not-infer-gene'}
 assert m.augment_groups(old,'UniProt',mapping)==['accession:P00001','detected:cluster','gene:17','sequence:exact']
 assert old==['accession:P00001','sequence:exact','gene:17']

def test_nucleotide_context_and_accession_only_references_do_not_acquire_protein_groups():
 assert m.augment_groups(['sequence:exact'],'Rfam15.1-native-reference-prior',{'exact':'detected:cluster'})==['sequence:exact']
 assert m.augment_groups(['accession:P00001'],'ReactZyme',{'P00001':'detected:cluster'})==['accession:P00001']
