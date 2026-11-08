"""Public-package integration of eight pure synthetic geometry leaves.

Fixtures retain exact caller axes; no biological/source/model authority is implied.
Historical setup error is tested as typed refusal, not dispatcher branch coverage.
"""
import dataclasses
import unittest
import numpy as np
from mora3.data import spatial_design_constraints as q
from mora3.geometry import rigid_constraints as m
from mora3.geometry import spatial_constraint_evaluation as v
from mora3.geometry import fixed_region_preservation as f
from mora3.geometry import structure_export as e
from mora3.geometry import reloaded_constraint_composition as base
from mora3.geometry import protein_reference_wire as p
from mora3.geometry import reloaded_reference_dispatch as api

def limits():
    return p.ReferenceBudget(8192,8192,32,8388608,8388608,25165824)

def finalize(req,candidate,references,records,scopes=(),identities=(),geometries=(),maxbytes=16777216):
    compiled=q.compile_design_request(req)
    scope=base.EvaluationScope(tuple(records),tuple(scopes),m.Numerics(1e-12,1e-10),maxbytes,tuple(identities),tuple(geometries))
    expected=tuple(api.ReferenceExpectation(x.profile,x.key,x.files.expectation,x.budget) for x in references)
    bound=api.declare_expectation(req,compiled,scope,candidate.expectation,expected)
    return dict(request=req,compiled=compiled,candidate_files=candidate,reference_files=tuple(references),scope=scope,expectation=bound)

def reb(c,**changes):
    c=dict(c,**changes)
    c['expectation']=api.declare_expectation(c['request'],c['compiled'],c['scope'],c['candidate_files'].expectation,
        tuple(api.ReferenceExpectation(x.profile,x.key,x.files.expectation,x.budget) for x in c['reference_files']))
    return c

def run(c):
    return api.evaluate_dispatched_constraints(**c,software=v.SoftwareBinding(v.REQUEST_SHA256,v.RIGID_SHA256,v.COMPILER_REVISION))

def template_case(weighted=False,evaluation=False,partial=False):
    seq=tuple(q.ResidueKey('S',i,'ALA') for i in range(2))
    tpl=tuple(q.ResidueKey('T',i+10,'MSE' if partial else 'ALA','ins') for i in range(2))
    out=tuple(q.AtomKey('enzyme',i,'C','CA',seq[i]) for i in range(2))
    ref=tuple(q.AtomKey('ref',i,'C','CA',tpl[i]) for i in range(2))
    ts=q.AtomKey('ts',0,'C','C')
    a=((-1.,-0.,0.),(1.,0.,0.));z=((-2.,-0.,0.),(2.,0.,0.)) if weighted else a
    candidate=e.export_bundle(e.ExportPacket('candidate','A',(e.ExportAtom(ts,(0.,0.,2.),e.Component.TS),)+tuple(e.ExportAtom(x,y,e.Component.ENZYME) for x,y in zip(out,a)),
        (e.SequenceChain('S','candidate-seq',seq,e.SequenceScope.FULL),)))
    purpose=p.ReferencePurpose.EVALUATION if evaluation else p.ReferencePurpose.GUIDANCE
    role=q.Role.EVALUATION_REFERENCE if evaluation else q.Role.USER_SUPPLIED
    pp=p.ProteinReferencePacket('original protein','B',tuple(e.ExportAtom(x,y,e.Component.ENZYME) for x,y in zip(ref,z)),
        (e.SequenceChain('T','original-seq',tuple(dataclasses.replace(x,identity='GLY') for x in tpl) if partial else tpl,e.SequenceScope.PARTIAL if partial else e.SequenceScope.FULL),),
        p.StructureScope.PARTIAL if partial else p.StructureScope.COMPLETE_SUPPLIED,purpose,role,context_json='{"source":"invented","unknown":null}')
    files=p.export_reference(pp,limits())
    cc=q.ConstraintSpec('enzyme-fit',q.Metric.ENZYME,1. if weighted else 0.,'ref',selection=q.Selection.BACKBONE,
        use=q.ConstraintUse.EVALUATION if evaluation else q.ConstraintUse.GUIDANCE,atoms=tuple(q.AtomPair(x,y) for x,y in zip(out,ref)),expected_reference_atoms=ref)
    req=q.DesignRequest(q.Direction.DOCK_KNOWN,(q.InputSpec('ts-in',q.InputKind.TS_GEOMETRY,q.Role.USER_SUPPLIED),
        q.InputSpec('seq',q.InputKind.ENZYME_SEQUENCE,q.Role.USER_SUPPLIED,residues=seq,complete=True),
        q.InputSpec('template',q.InputKind.ENZYME_STRUCTURE,q.Role.USER_SUPPLIED,residues=tpl)),q.Budget(1,128,128,256,1,1048576,1),
        references=(q.ReferenceSpec('ref',q.ReferenceKind.ENZYME_TEMPLATE,role,True,ref),),constraints=(cc,),
        residue_map=tuple(q.ResiduePair(x,y) for x,y in zip(seq,tpl)))
    records=(base.RecordBinding('enzyme','candidate','enzyme',out,v.Body.PROTEIN,v.Purpose.CANDIDATE_OUTPUT,q.Role.GENERATED_INTERMEDIATE,base.Use.OUTPUT),
        base.RecordBinding('ref','protein-ref','ref',ref,v.Body.PROTEIN,v.Purpose.EVALUATION_ONLY_REFERENCE if evaluation else v.Purpose.GUIDANCE_REFERENCE,role,base.Use.NUMERIC_REFERENCE))
    return finalize(req,candidate,(api.ReferenceFiles(api.Profile.PROTEIN_REFERENCE,'protein-ref',files,limits()),),records,
        (v.Scope('enzyme-fit',m.Coverage.PARTIAL_REFERENCE if partial else m.Coverage.FULL,(3.,1.) if weighted else (1.,1.)),))

def freeze_case(one_ulp=False,evaluation=False):
    c=template_case();req=c['request'];seq=req.inputs[1].residues;tpl=req.inputs[2].residues
    source=tuple(q.AtomKey('template',i,'C','CA',tpl[i]) for i in range(2))
    coords=((-1.,-0.,0.),(1.,0.,0.))
    purpose=p.ReferencePurpose.EVALUATION if evaluation else p.ReferencePurpose.GUIDANCE
    role=q.Role.EVALUATION_REFERENCE if evaluation else q.Role.USER_SUPPLIED
    pp=p.ProteinReferencePacket('supplied frozen template','A',tuple(e.ExportAtom(x,y,e.Component.ENZYME) for x,y in zip(source,coords)),(),
        p.StructureScope.COMPLETE_SUPPLIED,purpose,role)
    ref=api.ReferenceFiles(api.Profile.PROTEIN_REFERENCE,'frozen-input',p.export_reference(pp,limits()),limits())
    req=dataclasses.replace(req,references=(),constraints=(),regions=(q.RegionRule(tpl,q.InputPolicy.DEFAULT,freeze_sequence=True,freeze_all_atom_geometry=True),))
    packet=e.reload_bundle(c['candidate_files'],c['candidate_files'].expectation).packet
    if one_ulp:
        atoms=list(packet.atoms);atoms[2]=dataclasses.replace(atoms[2],xyz=(float(np.nextafter(1.,2.)),0.,0.))
        packet=dataclasses.replace(packet,atoms=tuple(atoms))
    candidate=e.export_bundle(packet)
    out=tuple(x.native for x in packet.atoms if x.component is e.Component.ENZYME)
    records=(base.RecordBinding('enzyme','candidate','enzyme',out,v.Body.PROTEIN,v.Purpose.CANDIDATE_OUTPUT,q.Role.GENERATED_INTERMEDIATE,base.Use.OUTPUT),
        base.RecordBinding('original','frozen-input','template',source,v.Body.PROTEIN,v.Purpose.GUIDANCE_REFERENCE,q.Role.USER_SUPPLIED,base.Use.FROZEN_SOURCE))
    ids=tuple(base.IdentityBinding(x,'candidate-seq',base.Slot(x.chain,x.index,x.insertion)) for x in seq)
    geo=tuple(base.GeometryBinding(tpl[i],'template','original','enzyme',(source[i],),(q.AtomPair(out[i],source[i]),),f.Coverage.COMPLETE_SUPPLIED_SUBJECT,f.FramePolicy.EXACT_COMMON_FRAME) for i in range(2))
    return finalize(req,candidate,(ref,),records,identities=ids,geometries=geo)

def mixed_pose_case():
    seq=tuple(q.ResidueKey('S',i,'ALA') for i in range(3));tpl=tuple(q.ResidueKey('T',i,'ALA') for i in range(3))
    xyz=((0.,0.,0.),(1.,0.,0.),(0.,1.,0.))
    ts=q.AtomKey('ts-out',0,'C','C');enz=tuple(q.AtomKey('enzyme',i,'C','CA',seq[i]) for i in range(3))
    rt=q.AtomKey('complex',0,'C','C');re=tuple(q.AtomKey('complex',i+1,'C','CA',tpl[i]) for i in range(3))
    original=tuple(q.AtomKey('template',i,'C','CA',tpl[i]) for i in range(3))
    chain=e.SequenceChain('S','candidate-seq',seq,e.SequenceScope.FULL)
    cand=e.export_bundle(e.ExportPacket('candidate','A',(e.ExportAtom(ts,(0.,0.,1.),e.Component.TS),)+tuple(e.ExportAtom(x,y,e.Component.ENZYME) for x,y in zip(enz,xyz)),(chain,)))
    complex_files=e.export_bundle(e.ExportPacket('bound-original','B',(e.ExportAtom(rt,(1.,0.,1.),e.Component.TS),)+tuple(e.ExportAtom(x,y,e.Component.ENZYME) for x,y in zip(re,xyz)),()))
    protein=p.export_reference(p.ProteinReferencePacket('protein-only-original','A',tuple(e.ExportAtom(x,y,e.Component.ENZYME) for x,y in zip(original,xyz)),(),
        p.StructureScope.COMPLETE_SUPPLIED,p.ReferencePurpose.GUIDANCE,q.Role.USER_SUPPLIED),limits())
    cc=q.ConstraintSpec('pose',q.Metric.TS_POSE,2.,'complex',alignment=q.Alignment.PROTEIN_ANCHORS,
        atoms=(q.AtomPair(ts,rt),),protein_anchors=tuple(q.AtomPair(x,y) for x,y in zip(enz,re)),expected_reference_atoms=(rt,))
    req=q.DesignRequest(q.Direction.DOCK_KNOWN,(q.InputSpec('visible-ts',q.InputKind.TS_GEOMETRY,q.Role.USER_SUPPLIED),
        q.InputSpec('seq',q.InputKind.ENZYME_SEQUENCE,q.Role.USER_SUPPLIED,residues=seq,complete=True),
        q.InputSpec('template',q.InputKind.ENZYME_STRUCTURE,q.Role.USER_SUPPLIED,residues=tpl)),q.Budget(1,128,128,256,1,1048576,1),
        references=(q.ReferenceSpec('complex',q.ReferenceKind.BOUND_COMPLEX,q.Role.USER_SUPPLIED,True,(rt,)+re),),constraints=(cc,),
        residue_map=tuple(q.ResiduePair(x,y) for x,y in zip(seq,tpl)),regions=(q.RegionRule(tpl,q.InputPolicy.DEFAULT,freeze_sequence=True,freeze_all_atom_geometry=True),))
    records=(base.RecordBinding('ts','candidate','ts-out',(ts,),v.Body.TS,v.Purpose.CANDIDATE_OUTPUT,q.Role.GENERATED_INTERMEDIATE,base.Use.OUTPUT),
        base.RecordBinding('enzyme','candidate','enzyme',enz,v.Body.PROTEIN,v.Purpose.CANDIDATE_OUTPUT,q.Role.GENERATED_INTERMEDIATE,base.Use.OUTPUT),
        base.RecordBinding('complex','bound','complex',(rt,)+re,v.Body.COMPLEX,v.Purpose.GUIDANCE_REFERENCE,q.Role.USER_SUPPLIED,base.Use.NUMERIC_REFERENCE),
        base.RecordBinding('frozen','protein','template',original,v.Body.PROTEIN,v.Purpose.GUIDANCE_REFERENCE,q.Role.USER_SUPPLIED,base.Use.FROZEN_SOURCE))
    ids=tuple(base.IdentityBinding(x,'candidate-seq',base.Slot(x.chain,x.index)) for x in seq)
    geo=tuple(base.GeometryBinding(tpl[i],'template','frozen','enzyme',(original[i],),(q.AtomPair(enz[i],original[i]),),f.Coverage.COMPLETE_SUPPLIED_SUBJECT,f.FramePolicy.EXACT_COMMON_FRAME) for i in range(3))
    refs=(api.ReferenceFiles(api.Profile.TS_STRUCTURE,'bound',complex_files,None),api.ReferenceFiles(api.Profile.PROTEIN_REFERENCE,'protein',protein,limits()))
    return finalize(req,cand,refs,records,(v.Scope('pose',m.Coverage.FULL,(1.,),m.Coverage.FULL,(1.,1.,1.)),),ids,geo)

class TestPublicDispatch(unittest.TestCase):
    def test_reference_only_template_no_dummy_ts(self):
        z=run(template_case());self.assertEqual(z.status,base.Status.DIAGNOSTIC_PASS)
        ref=z.artifacts[1].result.packet;self.assertTrue(all(x.component is e.Component.ENZYME for x in ref.atoms))
        self.assertIsInstance(z.artifacts[1].result,p.ReloadedReference);self.assertFalse(z.actual_export_acceptance)

    def test_independent_weighted_scale_oracle(self):
        z=run(template_case(weighted=True));self.assertAlmostEqual(z.numeric.results[0].evaluation.rmsd,(.75)**.5,places=13)
        self.assertEqual(z.status,base.Status.DIAGNOSTIC_PASS)

    def test_pure_protein_frozen_input_exact_literal(self):
        z=run(freeze_case());self.assertEqual(z.fixed.status,f.Status.PASS);self.assertEqual(z.status,base.Status.DIAGNOSTIC_PASS)
        self.assertIn('all_atom_freeze_unverified',z.numeric.policy_unknowns);self.assertFalse(z.numeric_adapter_hold_cleared)

    def test_one_ulp_frozen_violation_is_not_fit_pass(self):
        z=run(freeze_case(one_ulp=True));self.assertEqual(z.fixed.status,f.Status.VIOLATED);self.assertEqual(z.status,base.Status.VIOLATED_CANDIDATE)

    def test_evaluation_reference_is_posthoc_only(self):
        z=run(template_case(evaluation=True));self.assertEqual(z.status,base.Status.DIAGNOSTIC_PASS)
        self.assertEqual(z.numeric.predictor_reference_keys,())
        with self.assertRaises(RuntimeError):z.require_native_or_model_use()

    def test_evaluation_bytes_cannot_be_guidance(self):
        c=template_case(evaluation=True);a=list(c['scope'].records);a[1]=dataclasses.replace(a[1],purpose=v.Purpose.GUIDANCE_REFERENCE,role=q.Role.USER_SUPPLIED)
        c=reb(c,scope=dataclasses.replace(c['scope'],records=tuple(a)))
        with self.assertRaises(ValueError):run(c)

    def test_evaluation_bytes_cannot_be_frozen_source(self):
        with self.assertRaises(ValueError):run(freeze_case(evaluation=True))

    def test_explicit_profile_tag_rejects_wrong_wire_class(self):
        c=template_case();ref=c['reference_files'][0]
        with self.assertRaises(ValueError):api.ReferenceFiles(api.Profile.TS_STRUCTURE,ref.key,ref.files,None)
        with self.assertRaises(ValueError):api.ReferenceFiles(api.Profile.PROTEIN_REFERENCE,ref.key,c['candidate_files'],limits())

    def test_reference_budget_swap_refuses_original_binding(self):
        c=template_case();refs=c['reference_files'];c=dict(c,reference_files=(dataclasses.replace(refs[0],budget=dataclasses.replace(limits(),max_atoms=8191)),))
        with self.assertRaises(ValueError):run(c)

    def test_reference_expectation_swap_refuses(self):
        c=template_case();r=c['expectation'].references[0]
        bad=dataclasses.replace(r,expectation=dataclasses.replace(r.expectation,sidecar_sha256='0'*64))
        with self.assertRaises(ValueError):dataclasses.replace(c['expectation'],references=(bad,))

    def test_forged_compiled_plan_cannot_change_masks(self):
        c=template_case();c=dict(c,compiled=dataclasses.replace(c['compiled'],predictor_input_keys=('hidden',)))
        with self.assertRaises(ValueError):run(c)

    def test_changed_caller_threshold_refuses_bound_request(self):
        c=template_case();req=dataclasses.replace(c['request'],constraints=(dataclasses.replace(c['request'].constraints[0],threshold=2.),))
        with self.assertRaises(ValueError):run(dict(c,request=req,compiled=q.compile_design_request(req)))

    def test_changed_reference_frame_context_refuses_original(self):
        c=template_case();r=c['reference_files'][0];packet=p.reload_reference(r.files,r.files.expectation,budget=r.budget).packet
        changed=p.export_reference(dataclasses.replace(packet,frame='changed',context_json='{"different":true}'),r.budget)
        with self.assertRaises(ValueError):run(dict(c,reference_files=(dataclasses.replace(r,files=changed),)))

    def test_partial_modified_mismatch_remains_literal(self):
        z=run(template_case(partial=True));self.assertEqual(z.status,base.Status.UNSUPPORTED)
        p0=z.artifacts[1].result.packet
        self.assertEqual(p0.atoms[0].native.residue.identity,'MSE');self.assertEqual(p0.chains[0].residues[0].identity,'GLY')
        self.assertEqual(p0.structure_scope,p.StructureScope.PARTIAL)

    def test_decode_budget_exhaustion_has_no_numeric_result(self):
        c=template_case();c=reb(c,scope=dataclasses.replace(c['scope'],max_input_bytes=1))
        z=run(c);self.assertEqual(z.status,base.Status.BUDGET_EXHAUSTED);self.assertIsNone(z.numeric)

    def test_large_reference_never_clips_to_downstream128(self):
        c=template_case();r=c['reference_files'][0];packet=p.reload_reference(r.files,r.files.expectation,budget=r.budget).packet
        original=packet.atoms[0];atoms=tuple(dataclasses.replace(original,native=dataclasses.replace(original.native,ordinal=i,name='CA'+str(i))) for i in range(129))
        large=p.export_reference(dataclasses.replace(packet,atoms=atoms),r.budget)
        c=reb(c,reference_files=(dataclasses.replace(r,files=large),))
        z=run(c);self.assertEqual(z.status,base.Status.UNSUPPORTED);self.assertIsNone(z.numeric);self.assertEqual(len(z.artifacts[1].result.packet.atoms),129)

    def test_signed_zero_and_whole_namespace_selection(self):
        c=template_case();z=run(c);self.assertEqual(np.signbit(z.artifacts[1].result.packet.atoms[0].xyz[1]),True)
        records=list(c['scope'].records);records[1]=dataclasses.replace(records[1],atoms=records[1].atoms[:1])
        with self.assertRaises(ValueError):run(reb(c,scope=dataclasses.replace(c['scope'],records=tuple(records))))

    def test_legacy_pose_plus_protein_only_frozen_reference(self):
        z=run(mixed_pose_case());self.assertAlmostEqual(z.numeric.results[0].evaluation.rmsd,1.,places=13)
        self.assertEqual(z.fixed.status,f.Status.PASS);self.assertEqual(z.status,base.Status.DIAGNOSTIC_PASS)
        self.assertTrue(all(x.component is e.Component.ENZYME for x in z.artifacts[2].result.packet.atoms))

    def test_protein_reference_cannot_replace_candidate_packet(self):
        c=template_case();c=dict(c,candidate_files=c['reference_files'][0].files)
        with self.assertRaises(ValueError):run(c)

class TestPublicBoundaries(unittest.TestCase):
    def test_all_leaves_share_public_type_identity(self):
        self.assertIs(v.request_types, q)
        self.assertIs(f.request_types, q)
        self.assertIs(e.native_types, q)
        self.assertIs(base.numeric_leaf, v)
        self.assertIs(api.numeric_leaf, v)
        self.assertIs(p.wire, e)
        self.assertEqual(q.COMPILER_REVISION, 'Mora3.TSE.RequestRegionCompiler/v2')

    def test_whole_enzyme_pose_mapping_refuses_before_compilation(self):
        # Minimal exact witness for the historical eighteenth fixture error.
        # The downstream protein-profile TS-pose branch remains unqualified.
        ts = q.AtomKey('ts-output', 0, 'C', 'C')
        protein = q.AtomKey('protein-only', 3, 'C', 'CA', q.ResidueKey('T', 3, 'ALA'))
        with self.assertRaisesRegex(ValueError, 'unrelated protein/ligand atom sets'):
            q.AtomPair(ts, protein)

    def test_independent_external_expectation_survives_embedded_poison(self):
        c = template_case()
        files = c['candidate_files']
        c['candidate_files'] = dataclasses.replace(files, expectation=dataclasses.replace(files.expectation, sidecar_sha256='0'*64))
        self.assertEqual(run(c).status, base.Status.DIAGNOSTIC_PASS)

    def test_corrupt_wire_refuses_before_numeric(self):
        from unittest.mock import patch
        c = template_case()
        c['candidate_files'] = dataclasses.replace(c['candidate_files'], mmcif=c['candidate_files'].mmcif+b'\n')
        with patch.object(v, 'evaluate_request', side_effect=AssertionError('numeric reached corrupt wire')) as numeric:
            with self.assertRaises(ValueError):
                run(c)
            numeric.assert_not_called()

    def test_reordered_profiles_refuse_before_reload(self):
        from unittest.mock import patch
        c = mixed_pose_case()
        c['reference_files'] = tuple(reversed(c['reference_files']))
        with patch.object(e, 'reload_bundle', side_effect=AssertionError('decode reached reordered references')) as reload:
            with self.assertRaisesRegex(ValueError, 'complete ordered'):
                run(c)
            reload.assert_not_called()

    def test_numeric_pass_never_clears_source_model_export_holds(self):
        result = run(template_case())
        for method in (result.require_native_or_model_use, result.require_export_constraint_acceptance):
            with self.assertRaises(RuntimeError):
                method()
        self.assertEqual(result.authorization_status, 'HOLD')
        self.assertFalse(result.actual_export_acceptance)
        self.assertFalse(result.numeric_adapter_hold_cleared)
        self.assertIsNone(result.source_native_model_authority)
