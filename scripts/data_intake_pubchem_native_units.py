"""Cheap already-read native units; never infer neural token counts."""
from collections import Counter,defaultdict

class NativeUnitLedger:
    def __init__(self,snapshot=None):
        self.references=Counter();self.codecs=Counter();self.views=defaultdict(Counter)
        if snapshot is not None:
            if snapshot.get('schema')!='mora3.PubChemSerializedNativeUnitLedger/v1' or not snapshot.get('units_are_not_neural_tokens') or any(v is not None for v in snapshot['actual_model_token_totals'].values()):raise ValueError('native-only unit snapshot required')
            self.references.update(snapshot['unique_reference_graphs']);self.codecs.update(snapshot['unique_codec_graphs'])
            for direction,value in snapshot['per_view_direction'].items():self.views[direction].update(value)
    def reference_graph(self,row):
        self.references['graphs']+=1;self.references['native_SMILES_characters']+=len(row['native_smiles'])
        for field,out,missing in [('derived_atom_count','source_derived_atom_slots','source_atom_slots_missing'),('derived_bond_count','source_derived_bond_slots','source_bond_slots_missing')]:
            value=row.get(field)
            if type(value) is int and value>=0:self.references[out]+=value;self.references[out+'_observed_graphs']+=1
            else:self.references[missing]+=1
    def codec_graph(self,row):
        self._codec(row);self.codecs['graphs']+=1;self.codecs['standard_SELFIES_characters']+=len(row['standard_selfies']);self.codecs['standard_SELFIES_symbols']+=row['standard_selfies_token_count'];self.codecs['canonical_SMILES_characters']+=len(row['source_canonical_isomeric_smiles'])
    def _codec(self,row):
        if not isinstance(row.get('standard_selfies'),str) or not row['standard_selfies'] or type(row.get('standard_selfies_token_count')) is not int or row['standard_selfies_token_count']<1:raise ValueError('explicit eligible source SELFIES representation required')
    def view(self,direction,row):
        if direction not in ['reference_prior','smiles_to_selfies','selfies_to_smiles']:raise ValueError('explicit view direction required')
        if direction!='reference_prior':self._codec(row)
        value=self.views[direction]
        for key in ['input_SMILES_characters','input_standard_SELFIES_characters','input_standard_SELFIES_symbols','target_native_SMILES_characters','target_canonical_SMILES_characters','target_standard_SELFIES_characters','target_standard_SELFIES_symbols']:value[key]+=0
        value['views']+=1
        if direction=='reference_prior':value['target_native_SMILES_characters']+=len(row['native_smiles'])
        elif direction=='smiles_to_selfies':value['input_SMILES_characters']+=len(row['native_smiles']);value['target_standard_SELFIES_characters']+=len(row['standard_selfies']);value['target_standard_SELFIES_symbols']+=row['standard_selfies_token_count']
        else:value['input_standard_SELFIES_characters']+=len(row['standard_selfies']);value['input_standard_SELFIES_symbols']+=row['standard_selfies_token_count'];value['target_canonical_SMILES_characters']+=len(row['source_canonical_isomeric_smiles'])
    def snapshot(self):
        return dict(schema='mora3.PubChemSerializedNativeUnitLedger/v1',units_are_not_neural_tokens=True,unique_reference_graphs=dict(self.references),unique_codec_graphs=dict(self.codecs),per_view_direction={k:dict(v) for k,v in self.views.items()},actual_model_token_totals=dict(input=None,scored_target=None,padding=None,packed=None),model_reader_revision=None,model_tokenizer_revision=None,model_context_and_packing_revision=None,sampled_training_exposure=None,scope='serializedPubChemonly;verifieduniquegraphcallers;sourceatom/bondslotsnotTokenGTtokens;SELFIESsymbolsnotConvexTok/decoderunits;tasknames/fielddelimiters/IDs/auditexcluded;notanoverallcrossmodaltokentotal')
