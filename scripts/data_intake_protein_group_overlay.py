"""Detected protein groups are audit guards, never evidence or source-identity merges."""
PROTEIN_SOURCES=frozenset(['UniProt','UniProt-function-annotations','ReactZyme','UniProt-kinetics','UniProt-full-record-annotations','UniProt-GOA-assertion-packets'])
def augment_groups(original,source_family,canonical_sequence_groups):
    groups=set(original)
    if source_family in PROTEIN_SOURCES:
        for group in original:
            if group.startswith('sequence:'):
                detected=canonical_sequence_groups.get(group[len('sequence:'):])
                if detected is not None:groups.add(detected)
    return sorted(groups)
