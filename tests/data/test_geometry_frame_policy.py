"""Public metadata-only frame-policy contract and authority refusals."""
import dataclasses,unittest
from mora3.data import geometry_frame_policy as api
from mora3.data import spatial_design_constraints as q

def binding():
    return api.PolicyBinding(api.REQUEST_SHA256,api.COMPILER_REVISION,api.POLICY_SHA256)

def fixture(state=None,features=None,required=True,purpose=None,source_role=None,partial=False):
    a=tuple(q.AtomKey('ts',i,'C','C'+str(i)) for i in range(2))
    seq=(q.ResidueKey('S',0,'ALA'),)
    req=q.DesignRequest(q.Direction.DOCK_KNOWN,
        (q.InputSpec('ts',q.InputKind.TS_GEOMETRY,q.Role.USER_SUPPLIED),
         q.InputSpec('sequence',q.InputKind.ENZYME_SEQUENCE,q.Role.USER_SUPPLIED,residues=seq,complete=True),
         q.InputSpec('template',q.InputKind.ENZYME_STRUCTURE,q.Role.USER_SUPPLIED,residues=seq)),
        q.Budget(1,128,128,256,1,1048576,1),
        residue_map=(q.ResiduePair(seq[0],seq[0]),))
    compiled=q.compile_design_request(req)
    source=api.SourceDeclaration('synthetic:case','literal-v1','reader-v1','a'*64,'b'*64,'c'*64,None,
        api.Origin.SYNTHETIC_LITERAL,api.Purpose.CONDITION_INPUT if purpose is None else purpose)
    slot=api.SlotDeclaration(api.SlotName.TS_BODY,'ts',q.Role.USER_SUPPLIED if source_role is None else source_role,
        state or api.FrameState.INDEPENDENT_LOCAL,'angstrom','local',source,a,(False,False) if state is api.FrameState.UNAVAILABLE else (True,False) if partial else (True,True),
        api.axis_digest(a),'d'*64,api.MapOrigin.SYNTHETIC_EXPLICIT,api.POLICY_SHA256)
    current=api.CurrentFrame('template','current','angstrom',source)
    recipe=api.FrameRecipe('toy-policy',(api.SlotRecipe(api.SlotName.TS_BODY,required,features or (api.Feature.INTERNAL_DISTANCE,)),),('sequence',))
    return dict(request=req,compiled=compiled,recipe=recipe,current=current,slots=(slot,),budget=api.MetadataBudget(4,32768,256),software=binding())

def native(c,**kw):return api.compile_frame_policy(**c,**kw)
def synthetic(c,witnesses=()):
    return api.compile_synthetic_frame_policy(**c,witnesses=witnesses)

class FramePolicyTests(unittest.TestCase):
    def test_native_internal_is_descriptive_and_HOLD(self):
        z=native(fixture());self.assertEqual(z.status,api.Status.HOLD_AUTHORITY)
        self.assertEqual(z.slots[0].candidate_features,(api.Feature.INTERNAL_DISTANCE,))
        self.assertIsNone(z.actual_feature_arrays);self.assertIsNone(z.actual_source_authority)
    def test_independent_reference_cannot_request_mixed_basis(self):
        z=native(fixture(features=(api.Feature.MIXED_DOT,api.Feature.REFERENCE_VECTOR)))
        self.assertEqual(z.status,api.Status.UNSUPPORTED_FRAME)
        self.assertEqual(z.slots[0].candidate_features,())
    def test_synthetic_common_witness_is_not_native_registration(self):
        c=fixture(state=api.FrameState.REGISTERED_COMMON,features=(api.Feature.INTERNAL_DOT,api.Feature.MIXED_DOT))
        witness=api.declare_synthetic_witness(**c,slot_name=api.SlotName.TS_BODY)
        z=synthetic(c,(witness,));self.assertEqual(z.status,api.Status.SYNTHETIC_ELIGIBLE)
        self.assertIn(api.Feature.MIXED_DOT,z.slots[0].candidate_features)
        for method in (z.require_native_or_model_use,z.require_source_registration,z.require_loss_or_training,z.require_export_acceptance):
            with self.assertRaises(RuntimeError):method()
    def test_same_frame_strings_and_hash_claim_never_authorize(self):
        c=fixture(state=api.FrameState.REGISTERED_COMMON,features=(api.Feature.MIXED_DOT,))
        c=dict(c,current=dataclasses.replace(c['current'],frame_id='local'))
        self.assertEqual(native(c).status,api.Status.HOLD_AUTHORITY)
        self.assertEqual(synthetic(c).status,api.Status.HOLD_REGISTRATION)
    def test_required_unknown_and_unavailable_do_not_pass(self):
        for state in (api.FrameState.UNKNOWN,api.FrameState.UNAVAILABLE):
            c=fixture(state=state)
            z=native(c)
            self.assertIn(z.status,(api.Status.UNSUPPORTED_FRAME,api.Status.NOT_EVALUABLE))
            self.assertFalse(z.native_model_authorized)
    def test_optional_unavailable_slot_is_retained_not_dropped(self):
        z=native(fixture(state=api.FrameState.UNAVAILABLE,required=False))
        self.assertEqual(len(z.slots),1);self.assertEqual(z.slots[0].name,api.SlotName.TS_BODY)
        self.assertEqual(z.slots[0].candidate_features,())
    def test_evaluation_and_withheld_never_internal_predictors(self):
        for role,purpose in ((q.Role.EVALUATION_REFERENCE,api.Purpose.EVALUATION_REFERENCE),(q.Role.WITHHELD_TARGET,api.Purpose.WITHHELD_TARGET)):
            with self.assertRaises(api.FramePolicyError) as error:native(fixture(source_role=role,purpose=purpose))
            self.assertEqual(error.exception.code,'target_or_evaluation_forbidden')
    def test_changed_map_or_reader_invalidates_synthetic_witness(self):
        c=fixture(state=api.FrameState.REGISTERED_COMMON,features=(api.Feature.MIXED_DOT,))
        w=api.declare_synthetic_witness(**c,slot_name=api.SlotName.TS_BODY)
        for change in (dict(map_sha256='f'*64),dict(source=dataclasses.replace(c['slots'][0].source,reader_sha256='f'*64))):
            bad=dict(c,slots=(dataclasses.replace(c['slots'][0],**change),))
            with self.assertRaises(api.FramePolicyError):synthetic(bad,(w,))
    def test_forged_compiled_request_or_changed_raw_refuses(self):
        c=fixture();bad=dataclasses.replace(c['compiled'],predictor_input_keys=('hidden',))
        with self.assertRaises(api.FramePolicyError):native(dict(c,compiled=bad))
        changed=dataclasses.replace(c['request'],budget=dataclasses.replace(c['request'].budget,max_tokens=255))
        with self.assertRaises(api.FramePolicyError):native(dict(c,request=changed))
    def test_axis_hash_and_native_type_ambiguity_refuse(self):
        c=fixture()
        with self.assertRaises(api.FramePolicyError):dataclasses.replace(c['slots'][0],native_axis_sha256='0'*64)
        with self.assertRaises(ValueError):q.AtomKey('ts',True,'C','C')
    def test_whole_XYZ_boolean_mask_is_exact_and_partial_is_explicit(self):
        c=fixture()
        for mask in ((True,1),((True,True,True),(True,True,True))):
            with self.assertRaises(api.FramePolicyError):dataclasses.replace(c['slots'][0],xyz_available=mask)
        z=native(fixture(partial=True));self.assertEqual(z.status,api.Status.NOT_EVALUABLE)
        self.assertEqual(z.slots[0].xyz_available,(True,False));self.assertEqual(z.slots[0].candidate_features,())
    def test_recipe_slot_order_and_axis_duplicate_refuse(self):
        c=fixture();s=c['slots'][0]
        with self.assertRaises(api.FramePolicyError):native(dict(c,slots=()))
        with self.assertRaises(api.FramePolicyError):dataclasses.replace(s,native_axis=(s.native_axis[0],s.native_axis[0]))
    def test_fake_authority_never_inspected_or_accepted(self):
        class Fake:
            touched=False
            def __getattr__(self,name):self.touched=True;raise AssertionError('authority object inspected')
        c=fixture();fake=Fake()
        for value in (fake,True,False,{'approved':True},'a'*64):
            with self.assertRaises(api.FramePolicyError):native(c,registration_authority=value)
        self.assertFalse(fake.touched)
    def test_wrong_policy_or_compiler_binding_refuses(self):
        for values in ((api.REQUEST_SHA256,'old',api.POLICY_SHA256),(api.REQUEST_SHA256,api.COMPILER_REVISION,'0'*64)):
            with self.assertRaises(api.FramePolicyError):api.PolicyBinding(*values)
    def test_target_map_and_old_RMSD_alignment_are_forbidden(self):
        c=fixture()
        for provenance in (api.MapOrigin.TARGET_DERIVED,api.MapOrigin.POSTHOC_RMSD,api.MapOrigin.GOLD_ALIGNMENT):
            with self.assertRaises(api.FramePolicyError) as error:native(dict(c,slots=(dataclasses.replace(c['slots'][0],map_origin=provenance),)))
            self.assertEqual(error.exception.code,'target_derived_registration_forbidden')
    def test_no_callbacks_or_mutation_or_partial_count_authority(self):
        c=fixture();before=c['compiled'].canonical_bytes()
        called=[]
        with self.assertRaises(TypeError):native(c,geometry_callback=lambda:called.append(True))
        z=native(c);self.assertEqual(called,[]);self.assertEqual(c['compiled'].canonical_bytes(),before)
        with self.assertRaises(dataclasses.FrozenInstanceError):z.status=api.Status.SYNTHETIC_ELIGIBLE
        self.assertIsNone(z.actual_model_token_positions);self.assertFalse(z.fullready)

class PublicBoundaryTests(unittest.TestCase):
    def test_public_request_types_have_single_identity(self):
        self.assertIs(api.request_types, q)
        self.assertIs(type(fixture()['request']), q.DesignRequest)

    def test_predictor_reference_stays_metadata_only(self):
        c=fixture()
        reference=q.ReferenceSpec('reference',q.ReferenceKind.TS,q.Role.USER_SUPPLIED,True,
            (q.AtomKey('reference',0,'C','C0'),))
        constraint=q.ConstraintSpec('shape',q.Metric.TS_SHAPE,0.,'reference',
            atoms=(q.AtomPair(c['slots'][0].native_axis[0],reference.atoms[0]),),
            expected_reference_atoms=reference.atoms)
        request=dataclasses.replace(c['request'],references=(reference,),constraints=(constraint,))
        compiled=q.compile_design_request(request)
        self.assertEqual(compiled.predictor_reference_keys,('reference',))
        z=native(dict(c,request=request,compiled=compiled))
        self.assertEqual(z.status,api.Status.HOLD_AUTHORITY)
        self.assertEqual(z.slots[0].native_axis,c['slots'][0].native_axis)
        self.assertIsNone(z.actual_feature_arrays)
        for method in (z.require_native_or_model_use,z.require_source_registration,
                       z.require_loss_or_training,z.require_export_acceptance):
            with self.assertRaises(RuntimeError):method()

    def test_every_authority_argument_refuses_without_inspection(self):
        class Uninspectable:
            def __getattribute__(self,name):raise AssertionError('inspected authority')
        for name in ('registration_authority','source_authority','model_authority'):
            with self.subTest(name=name),self.assertRaises(api.FramePolicyError) as caught:
                native(fixture(),**{name:Uninspectable()})
            self.assertEqual(caught.exception.code,'authority_bridge_unimplemented_no_caller_grant')

    def test_native_source_claim_does_not_become_synthetic_evidence(self):
        c=fixture()
        source=dataclasses.replace(c['slots'][0].source,origin=api.Origin.UNQUALIFIED_SOURCE,source_id='unqualified:source')
        c=dict(c,slots=(dataclasses.replace(c['slots'][0],source=source),))
        self.assertFalse(native(c).native_model_authorized)
        with self.assertRaises(api.FramePolicyError) as caught:synthetic(c)
        self.assertEqual(caught.exception.code,'synthetic_diagnostic_requires_only_literal_declarations')
