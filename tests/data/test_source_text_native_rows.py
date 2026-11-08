from dataclasses import replace
import hashlib,json
from pathlib import Path
import tempfile
import unittest
from mora3.data.source_text_native_rows import (wiki_rows,sefaria_row,logical_character_slots,
    oshb_book_row,reasoning_row,jats_row,native_word_root_context,SourceTextNativeBinding,FAMILIES)


def node(text=None,tail=None,children=None,tag='native'):
    return dict(tag=tag,attributes={},text=text,tail=tail,children=[] if children is None else children)


def wiki(text='א\u05b0<source>{{inert}}</source>'):
    return dict(native_page_id=1,text_normalization_performed=False,native_revisions=[dict(
        native_revision_id=2,native_text_exact=text,native_text_UTF8_bytes=len(text.encode()),
        native_text_available=True,native_declared_byte_count_matches=True,native_declared_base36_SHA1_matches=True)])


class ExactNativeTextTests(unittest.TestCase):
    def test_wiki_exact_full_codepoints_markup_and_unknown_role(self):
        source=wiki();row=next(wiki_rows(source,dict(source='native')))
        self.assertEqual(row.full_target_utf8,source['native_revisions'][0]['native_text_exact'].encode())
        self.assertIsNone(row.original_role)
        self.assertFalse(row.training_authorized)
        self.assertEqual(row.condition_inputs(),dict(task='source_language_prior',language='he',domain='biomedical'))
        self.assertNotIn('native_page_id',row.condition_inputs())

    def test_missing_or_changed_revision_text_is_not_silently_accepted(self):
        for key,value in [('native_text_available',False),('native_text_UTF8_bytes',1),
                ('native_declared_base36_SHA1_matches',False)]:
            source=wiki();source['native_revisions'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):next(wiki_rows(source,{}))

    def test_sefaria_keeps_html_and_combining_marks_without_verse_zip(self):
        text='<b>א\u05b0</b>\n'
        native=dict(text_exact_native=text,text_SHA256=hashlib.sha256(text.encode()).hexdigest(),
            text_normalization_performed=False,native_license='Public Domain',source_book_group_key='native-book')
        row=sefaria_row(native,{})
        self.assertEqual(row.full_target_utf8,text.encode())
        self.assertEqual(row.group_keys[0],'source-lineage:http://www.tanach.us/Tanach.xml')
        self.assertIsNone(row.original_role)

    def test_complete_text_tail_order_has_no_inserted_separators(self):
        tree=node('A',children=[node('B','C'),node('', 'D')])
        slots=list(logical_character_slots(tree))
        self.assertEqual([s['text_exact_native'] for s in slots],['A','B','C','','D'])
        self.assertEqual([s['source_child_ordinal_path'] for s in slots],[[],[0],[0],[1],[1]])
        row=oshb_book_row(tree,{},'NativeOSIS')
        self.assertEqual(row.full_target_utf8,b'ABCD')
        self.assertNotIn('native_book_OSIS_ID',row.condition_inputs())

    def jats(self):
        tree=node('Title',children=[node(' exact',' tail')])
        segments=list(logical_character_slots(tree))
        return dict(complete_XML_logical_tree=tree,native_character_data_segments=segments,
            XML_codepoints_or_source_text_normalized=False,qualification=dict(native_article_ID_fields=[
                dict(native_type='pmcid',native_value='PMC1'),dict(native_type='pmid',native_value='2')]),
            bibliography=[dict(PMID='999')])

    def test_article_uses_all_slots_and_only_self_ids_for_groups(self):
        source=self.jats();row=jats_row(source,{})
        self.assertEqual(row.full_target_utf8,b'Title exact tail')
        self.assertEqual(set(row.group_keys),{'article-self-pmcid:PMC1','article-self-pmid:2'})
        self.assertNotIn('999',' '.join(row.group_keys))
        self.assertEqual(row.rights_disposition,'unreviewed-original-grant-context')

    def test_article_segment_omission_or_reordering_refuses(self):
        for change in ('omit','reorder'):
            source=self.jats()
            if change=='omit':source['native_character_data_segments'].pop()
            else:source['native_character_data_segments'].reverse()
            with self.subTest(change=change),self.assertRaises(ValueError):jats_row(source,{})

    def test_gsm_full_answer_and_exact_question_group_preserve_source_role(self):
        native=dict(source_repository='openai/grade-school-math',official_role='train',
            native_fields=dict(question='Question?\n',answer='Full reasoning\n#### 2'))
        first=reasoning_row(native,dict(variant='main'))
        second=reasoning_row(dict(native,native_fields=dict(question='Question?\n',answer='Socratic\n#### 2')),dict(variant='socratic'))
        self.assertEqual(first.full_target_utf8,b'Full reasoning\n#### 2')
        self.assertEqual(first.group_keys,second.group_keys)
        self.assertEqual(first.original_role,'train')
        self.assertEqual(set(first.condition_inputs()),{'task','question'})

    def humaneval(self):
        return dict(source_repository='openai/human-eval',official_role='evaluation_only',
            generated_code_or_tests_executed=False,native_fields=dict(task_id='HumanEval/0',prompt='def f():\n',
                canonical_solution='    raise RuntimeError("inert")\n',test='raise RuntimeError("test remains inert")',entry_point='f'))

    def test_humaneval_solution_is_evaluation_only_with_test_and_entrypoint_audit_only(self):
        native=self.humaneval();row=reasoning_row(native,{})
        self.assertTrue(row.evaluation_only)
        self.assertEqual(row.original_role,'evaluation_only')
        self.assertEqual(row.full_target_utf8,native['native_fields']['canonical_solution'].encode())
        self.assertEqual(row.condition_inputs(),dict(task='humaneval_solution',prompt='def f():\n'))
        self.assertEqual(row.source_audit()['native_fields']['test'],native['native_fields']['test'])

    def test_humaneval_training_or_executed_code_flags_refuse(self):
        for key,value in [('official_role','train'),('generated_code_or_tests_executed',True)]:
            native=self.humaneval();native[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):reasoning_row(native,{})

    def root_word(self):
        return dict(native_word_ID='nativeword',native_word_direct_text='אָב',source_book_OSIS_ID='nativebook',
            source_child_ordinal_path=[0,1],native_lemma='gold-lemma',native_morph='gold-morph',
            direct_original_lexicon_root_bindings=[dict(component_ordinal=0,native_aug_code='goldcode',
                lexical_entry_id='goldlexicon',native_root='אבגד',native_lexical_entry_ordinal=1)])

    def test_root_targets_preserve_literal_nontriconsonantal_source_assertion(self):
        row=native_word_root_context(self.root_word(),{})
        self.assertEqual(row.unknown_root_observations(),{'observed_word':'אָב'})
        roots=json.loads(row.direct_root_assertions_json)
        self.assertEqual(roots[0]['native_root'],'אבגד')
        self.assertEqual(row.root_target_status,'observed-direct-source-assertions')
        self.assertFalse(row.training_authorized)
        self.assertNotIn('gold-lemma',json.dumps(row.unknown_root_observations()))

    def test_absent_direct_root_is_unlabeled_never_empty_gold_or_guessed(self):
        source=self.root_word();source['direct_original_lexicon_root_bindings']=[]
        row=native_word_root_context(source,{})
        self.assertEqual(row.root_target_status,'unlabeled')
        self.assertEqual(json.loads(row.direct_root_assertions_json),[])

    def test_missing_native_root_or_malformed_coordinate_refuses(self):
        source=self.root_word();source['direct_original_lexicon_root_bindings'][0]['native_root']=None
        with self.assertRaises(ValueError):native_word_root_context(source,{})
        source=self.root_word();source['source_child_ordinal_path']=[True]
        with self.assertRaises(ValueError):native_word_root_context(source,{})


class PinnedBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.manifest=self.put('source-manifest.json',b'{"source":"inert fixture"}')
        doc=dict(families=[dict(id=f,manifest=self.manifest,**(dict(audit_only_grant_conflicts=[],articles=[]) if f=='INV-083' else {})) for f in FAMILIES],
            historical7_candidate_count_superseded_by6_consistent_declarations_and1_conflict=True)
        self.document=doc;self.handoff=self.put('handoff.json',json.dumps(doc).encode())

    def put(self,name,raw):
        p=self.root/name;p.write_bytes(raw);return dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

    def binding(self):
        return SourceTextNativeBinding(self.root/'handoff.json',self.handoff['sha256'],root=self.root)

    def test_actual_pinned_manifest_identity_changes_refuse(self):
        binding=self.binding();(self.root/'source-manifest.json').write_bytes(b'changed')
        with self.assertRaises(ValueError):binding.require_sources_unchanged()

    def test_untrusted_handoff_or_legacy_grant_descriptor_refuses(self):
        with self.assertRaises(ValueError):SourceTextNativeBinding(self.root/'handoff.json','0'*64,root=self.root)
        self.document.pop('historical7_candidate_count_superseded_by6_consistent_declarations_and1_conflict')
        self.handoff=self.put('handoff.json',json.dumps(self.document).encode())
        with self.assertRaises(ValueError):self.binding()

    def test_unknown_or_duplicate_source_selection_has_no_fallback(self):
        binding=self.binding()
        for families in ((),('unknown',),('INV-083','INV-083')):
            with self.subTest(families=families),self.assertRaises(ValueError):list(binding.iter_rows(families=families))


class JATSNontextSourceGrammarTests(unittest.TestCase):
    def test_processing_instruction_and_comment_contents_remain_audit_but_tails_are_targets(self):
        tree=node('A',children=[node('inert instruction','B',tag='#processing-instruction'),
            node('native comment','C',tag='#comment'),node('D','E',tag='bold')])
        expected=[dict(source_child_ordinal_path=[],slot='text',text_exact_native='A'),
            dict(source_child_ordinal_path=[0],slot='tail',text_exact_native='B'),
            dict(source_child_ordinal_path=[1],slot='tail',text_exact_native='C'),
            dict(source_child_ordinal_path=[2],slot='text',text_exact_native='D'),
            dict(source_child_ordinal_path=[2],slot='tail',text_exact_native='E')]
        native=dict(complete_XML_logical_tree=tree,native_character_data_segments=expected,
            XML_codepoints_or_source_text_normalized=False,
            qualification=dict(native_article_ID_fields=[dict(native_type='pmcid',native_value='PMC1')]))
        row=jats_row(native,{})
        self.assertEqual(row.full_target_utf8,b'ABCDE')
        self.assertEqual(row.source_audit()['complete_XML_logical_tree'],tree)
        self.assertEqual(row.source_audit()['complete_XML_logical_tree']['children'][0]['text'],'inert instruction')

    def test_nontext_target_insertion_does_not_match_declared_native_slots(self):
        tree=node('A',children=[node('PI data','B',tag='#processing-instruction')])
        native=dict(complete_XML_logical_tree=tree,
            native_character_data_segments=list(logical_character_slots(tree)),
            XML_codepoints_or_source_text_normalized=False,
            qualification=dict(native_article_ID_fields=[dict(native_type='pmcid',native_value='PMC1')]))
        with self.assertRaises(ValueError):jats_row(native,{})


class ObservedRootListViewTests(unittest.TestCase):
    root_word=ExactNativeTextTests.root_word

    def context(self,source=None,subtree=None):
        source=self.root_word() if source is None else source
        source['native_word_attributes_JSON']='{}'
        tree=node(source['native_word_direct_text'],tag='{native-osis}w') if subtree is None else subtree
        return native_word_root_context(source,dict(source='exact-native-fixture'),source_word_subtree=tree)

    def test_native_literal_root_strings_not_IDs_form_explicit_UTF8_JSON_array(self):
        source=self.root_word();source['direct_original_lexicon_root_bindings']*=2
        context=self.context(source);view=context.observed_root_list_view()
        self.assertEqual(view.condition_inputs(),{'task':'oshb_source_root_list','word':'אָב'})
        self.assertEqual(json.loads(view.full_target_utf8),['אבגד','אבגד'])
        self.assertNotIn(b'goldlexicon',view.full_target_utf8)
        self.assertEqual(view.group_keys[0],'source-lineage:http://www.tanach.us/Tanach.xml')
        self.assertFalse(view.training_authorized)

    def test_unknown_root_has_no_view_or_empty_EOS_supervision(self):
        source=self.root_word();source['direct_original_lexicon_root_bindings']=[]
        self.assertIsNone(self.context(source).observed_root_list_view())

    def test_child_bearing_word_is_explicitly_unqualified_not_silently_cropped(self):
        source=self.root_word();tree=node(source['native_word_direct_text'],children=[node('native extra')],tag='w')
        context=self.context(source,tree)
        self.assertEqual(context.word_input_status,'native-word-has-child-content')
        with self.assertRaises(ValueError):context.observed_root_list_view()
        self.assertEqual(context.source_audit_json.count(b'native extra'),1)

    def test_bare_direct_slot_cannot_self_claim_complete_word(self):
        context=native_word_root_context(self.root_word(),{})
        with self.assertRaises(ValueError):context.observed_root_list_view()

    def test_wrong_original_word_subtree_text_or_gold_attribute_binding_refuses(self):
        for tree in (node('wrong',tag='w'),node('אָב',tag='other')):
            with self.subTest(tree=tree),self.assertRaises(ValueError):self.context(subtree=tree)


class NativeInlineFormattingWordTests(unittest.TestCase):
    root_word=ExactNativeTextTests.root_word

    def context(self,kind='x-large',tag='seg',extra=False):
        source=self.root_word();source['native_word_attributes_JSON']='{}'
        child=node('ג','ד',tag=tag);child['attributes']={'type':kind}
        if extra:child['attributes']['unknown']='not-glyph-only'
        tree=node('אָב',tail='outside-parent-word',children=[child],tag='w')
        return native_word_root_context(source,{},source_word_subtree=tree)

    def test_exact_inline_glyph_characters_and_tails_form_full_word_without_parent_tail(self):
        for kind in ('x-large','x-small','x-suspended'):
            context=self.context(kind)
            self.assertEqual(context.observed_word_utf8,'אָבגד'.encode())
            self.assertEqual(context.word_input_status,'complete-native-word-inline-formatting')
            view=context.observed_root_list_view()
            self.assertEqual(view.condition_inputs(),{'task':'oshb_source_root_list','word':'אָבגד'})
            self.assertEqual(json.loads(view.full_target_utf8),['אבגד'])
            self.assertIn('outside-parent-word',context.source_audit_json.decode())

    def test_note_alternative_reading_or_unknown_attributes_never_get_flattened(self):
        for kwargs in (dict(tag='note'),dict(kind='alternative-reading'),dict(extra=True)):
            context=self.context(**kwargs)
            self.assertEqual(context.word_input_status,'native-word-has-child-content')
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):context.observed_root_list_view()


if __name__=='__main__':unittest.main()
