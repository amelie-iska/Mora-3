"""Fit-bound scCello sparse packets with visible-count-only normalization.

The new raw-count token coordinate is separate from the legacy provider-export
coordinate. This module prepares inactive packets and grants no data roles.
"""
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import re
from types import MappingProxyType

FIT_SCHEMA="Mora3.InactiveRawFitPackedExact/v1"
RECEIPT_SCHEMA="Mora3.InactiveMacExactRawFitTerminalReceipt/v1"
RESERVED={"PAD":0,"MASK":1,"CLS":2}
_SHA=re.compile(r"[a-f0-9]{64}\Z")
MAX_METADATA_BYTES=16*1024**2


def require(ok,message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()


def pinned(root,path,expected):
    candidate=Path(path)
    require(type(expected) is str and _SHA.fullmatch(expected) is not None and candidate.is_absolute(),
            "exact absolute scoped metadata pin required")
    require(candidate.is_relative_to(root) and not any(p.is_symlink() for p in (candidate,*candidate.parents)),
            "metadata path outside private scope or symlink")
    require(candidate.is_file() and candidate.stat().st_size<=MAX_METADATA_BYTES,
            "bounded regular metadata required")
    with candidate.open("rb") as stream:
        raw=stream.read(MAX_METADATA_BYTES+1)
    require(len(raw)<=MAX_METADATA_BYTES and sha256(raw).hexdigest()==expected,"metadata pin changed")
    return json.loads(raw)


@dataclass(frozen=True,slots=True)
class RawFitAxis:
    fit_sha256: str
    evidence_sha256: str
    evidence_kind: str
    gene_keys: tuple[str,...]
    median_rational: tuple[tuple[int,int],...]
    _tokens: object

    def __deepcopy__(self, memo):
        # The fitted gene and median tuples are immutable. Give the staged
        # owner a separate read-only token index and preserve aliases in memo.
        result = type(self)(self.fit_sha256, self.evidence_sha256, self.evidence_kind,
                            self.gene_keys, self.median_rational,
                            MappingProxyType(dict(self._tokens)))
        memo[id(self)] = result
        return result

    @classmethod
    def _from_products(cls,fit,receipt,*,fit_sha256,terminal_sha256):
        require(type(receipt) is dict and receipt.get("schema")==RECEIPT_SCHEMA
                and receipt.get("final_fit_SHA256")==fit_sha256
                and receipt.get("source_role")=="inactive_reference" and receipt.get("training_authorized") is False
                and receipt.get("all_median_and_vocabulary_records_independently_recomputed") is True
                and receipt.get("no_validation_objects") is True and receipt.get("no_old_export_join") is True
                and receipt.get("fit_sources")==4 and receipt.get("exact_rows")==360501,
                "completed independent train-only fit terminal receipt required")
        require(all(type(x) is str and _SHA.fullmatch(x) is not None for x in (fit_sha256,terminal_sha256)),
                "exact fit and terminal SHA256 required")
        return cls._from_fit(fit,receipt,fit_sha256,terminal_sha256,"original-terminal-readback")

    @classmethod
    def _from_fit(cls,fit,receipt,fit_sha256,evidence_sha256,evidence_kind):
        require(type(fit) is dict and set(fit)=={"schema","metrics","genes","reserved","source_role","training_authorized"}
                and fit["schema"]==FIT_SCHEMA and fit["reserved"]==RESERVED and fit["source_role"]=="inactive_reference"
                and fit["training_authorized"] is False and type(fit["metrics"]) is dict,
                "exact native raw-fit product required")
        metrics=fit["metrics"]
        require(set(metrics)=={"rows","positive_entries","empty_rows","explicit_zeros"}
                and all(type(v) is int and v>=0 for v in metrics.values())
                and metrics["rows"]==receipt["exact_rows"]
                and all(metrics[k]==receipt[k] for k in ("positive_entries","empty_rows","explicit_zeros"))
                and metrics["positive_entries"]+metrics["explicit_zeros"]==receipt["exact_stored_sparse_entries"],
                "fit and independent terminal support denominators differ")
        genes=fit["genes"]
        require(type(genes) is list and 0<len(genes)==receipt["scoped_gene_tokens"],
                "exact fitted finite vocabulary extent required")
        keys=[]
        medians=[]
        supports=0
        for token,gene in enumerate(genes,3):
            require(type(gene) is dict and set(gene)=={"gene","token","support","median"}
                    and gene["token"]==token and type(gene["token"]) is int and type(gene["gene"]) is str
                    and 0<len(gene["gene"])<=256 and gene["gene"] not in RESERVED and
                    (not keys or keys[-1].encode("utf-8")<gene["gene"].encode("utf-8"))
                    and type(gene["support"]) is int and 0<gene["support"]<=metrics["rows"],
                    "raw-fit gene key, ordering, token or support differs")
            median=gene["median"]
            require(type(median) is list and len(median)==2 and all(type(v) is int and v>0 for v in median),
                    "exact positive median rational required")
            rational=Fraction(*median)
            require(rational<=10000 and (rational.numerator,rational.denominator)==tuple(median),
                    "reduced native median rational outside source normalization")
            keys.append(gene["gene"])
            medians.append(tuple(median))
            supports+=gene["support"]
        require(supports==metrics["positive_entries"],"fitted gene support does not close")
        return cls(fit_sha256,evidence_sha256,evidence_kind,tuple(keys),tuple(medians),
                   MappingProxyType({name:token for token,name in enumerate(keys,3)}))

    @classmethod
    def load(cls,private_root,terminal_path,*,terminal_sha256):
        root=Path(private_root).resolve()
        receipt=pinned(root,terminal_path,terminal_sha256)
        fit_sha=receipt.get("final_fit_SHA256")
        fit=pinned(root,receipt.get("final_fit_path",""),fit_sha)
        return cls._from_products(fit,receipt,fit_sha256=fit_sha,terminal_sha256=terminal_sha256)

    @classmethod
    def load_native_dual(cls,private_root,fit_path,selection_path,fullsort_path,*,
                         selection_sha256,fullsort_sha256):
        root=Path(private_root).resolve()
        selection=pinned(root,selection_path,selection_sha256)
        fullsort=pinned(root,fullsort_path,fullsort_sha256)
        require(selection.get("schema")=="Mora3.InactiveScCelloNativeIndependentMedianCandidate/v1"
                and selection.get("exact_native_ratio_selection") is True
                and selection.get("original_sources_preserved") is True
                and selection.get("fit_sources")==4 and selection.get("exact_rows")==360501
                and selection.get("validation_objects_read") is False and selection.get("training_authorized") is False
                and type(selection.get("source_roles_assigned")) is int and selection["source_roles_assigned"]==0,
                "exact inactive native selection proof required")
        require(fullsort.get("schema")=="Mora3.ScCelloIndependentCFullSortReadback/v1"
                and fullsort.get("verified") is True and fullsort.get("exact_candidate_SHA_reproduced") is True
                and fullsort.get("source_blocks_unchanged") is True and fullsort.get("candidate_unchanged") is True
                and fullsort.get("validation_objects_read") is False and fullsort.get("source_roles_changed") is False
                and fullsort.get("training_authorized") is False
                and fullsort.get("cells")==selection["exact_rows"] and fullsort.get("genes")==selection.get("genes")
                and fullsort.get("positive_entries")==selection.get("positive_entries")
                and fullsort.get("blocks")==selection.get("blocks") and fullsort.get("mapped_bytes")==selection.get("original_pack_bytes")
                and fullsort.get("candidate_sha256")==selection.get("median_candidate_sha256"),
                "separate complete native full-sort proof must match selection")
        fit_sha=selection["median_candidate_sha256"]
        fit=pinned(root,fit_path,fit_sha)
        require(type(fit) is dict and type(fit.get("metrics")) is dict,"exact fitted metrics required")
        metrics=fit["metrics"]
        require(type(metrics.get("explicit_zeros")) is int and type(metrics.get("empty_rows")) is int,
                "exact native zero and empty row denominators required")
        counts=dict(exact_rows=selection["exact_rows"],positive_entries=selection["positive_entries"],
                    empty_rows=metrics["empty_rows"],explicit_zeros=metrics["explicit_zeros"],
                    exact_stored_sparse_entries=selection["positive_entries"]+metrics["explicit_zeros"],
                    scoped_gene_tokens=selection["genes"])
        chain=sha256(canonical(("raw-fit-dual-native-v1",fit_sha,selection_sha256,fullsort_sha256))).hexdigest()
        return cls._from_fit(fit,counts,fit_sha,chain,"native-selection-and-independent-fullsort")


@dataclass(frozen=True,slots=True)
class BoundRawAxis:
    vocabulary: RawFitAxis
    original_axis_sha256: str
    _native_axis: tuple[str,...]
    _tokens: tuple[int|None,...]

    @classmethod
    def bind(cls,vocabulary,original_axis,*,expected_axis_sha256):
        require(type(vocabulary) is RawFitAxis and type(original_axis) is tuple and original_axis
                and all(type(name) is str and name for name in original_axis)
                and len(set(original_axis))==len(original_axis),
                "exact immutable original CSR feature axis required")
        require(sha256(canonical(original_axis)).hexdigest()==expected_axis_sha256,
                "raw CSR feature axis digest differs")
        return cls(vocabulary,expected_axis_sha256,original_axis,
                   tuple(vocabulary._tokens.get(name) for name in original_axis))


@dataclass(frozen=True,slots=True)
class RawMaskedPacket:
    visible: tuple[tuple[int,Fraction],...]
    targets: tuple[tuple[int,Fraction],...]
    visible_library_count: int
    fitted_axis_evidence_sha256: str


def prepare_masked_packet(axis,stored_entries,visible_columns,target_columns):
    """Use the same visible-only library scale for input and target labels."""
    require(type(axis) is BoundRawAxis and type(stored_entries) is tuple
            and type(visible_columns) is frozenset and type(target_columns) is frozenset
            and all(type(i) is int for i in visible_columns|target_columns)
            and visible_columns.isdisjoint(target_columns) and target_columns,
            "explicit disjoint source-column masks with observed targets required")
    counts={}
    for entry in stored_entries:
        require(type(entry) is tuple and len(entry)==2 and all(type(v) is int for v in entry),
                "exact original integral CSR pair required")
        column,count=entry
        require(0<=column<len(axis._tokens) and column not in counts and 0<=count<=2**24,
                "duplicate or invalid source count coordinate")
        counts[column]=count
    selected=visible_columns|target_columns
    require(selected<=counts.keys(),"no implicit or absent-gene targets")
    require(all(axis._tokens[column] is not None for column in selected),
            "selected original gene absent from the independently fitted axis")
    total=sum(counts[column] for column in visible_columns)
    require(total>0,"positive visible-only library size required")
    def assemble(mask):
        return tuple((axis._tokens[i],Fraction(10000*count,total))
                     for i,count in counts.items() if i in mask)
    return RawMaskedPacket(assemble(visible_columns),assemble(target_columns),total,axis.vocabulary.evidence_sha256)
