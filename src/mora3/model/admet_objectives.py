"""Endpoint-local mathematical ADMET objectives, not an admission or universal score.

Callers must retain native units/readout/assay partitions. These tensor kernels
do not grant observed-label eligibility. Actual ADMET supervision remains gated
until a verified source-specific admitted task reader is integrated.
"""
from dataclasses import dataclass
import importlib
import math

import torch
from torch.nn import functional as F

data = importlib.import_module("mora3.data.admet")
_LOG2PI = math.log(2*math.pi)


def _reference(x):
    if not isinstance(x,torch.Tensor) or x.ndim!=1 or x.dtype not in (torch.float32,torch.float64):
        raise ValueError("one endpoint's explicit float32/float64 prediction vector required")


def _numeric(reference,value,name):
    if (not isinstance(value,torch.Tensor) or value.shape!=reference.shape
            or value.dtype!=reference.dtype or value.device!=reference.device):
        raise ValueError(name+" must preserve explicit loss shape/device/precision")


def _active(reference,observed,applicable,applicability_known):
    _reference(reference)
    for value in (observed,applicable,applicability_known):
        if (not isinstance(value,torch.Tensor) or value.dtype!=torch.bool
                or value.shape!=reference.shape or value.device!=reference.device):
            raise ValueError("separate exact boolean observation/applicability/known masks required")
    return observed & applicable & applicability_known


def _finite(value):
    if not torch.isfinite(value).all():raise ValueError("active native targets/predictions must be finite")


def masked_binary_bce(logits,target,*,observed,applicable,applicability_known):
    """Native binary labels only; never threshold a continuous readout."""
    active=_active(logits,observed,applicable,applicability_known)
    _numeric(logits,target,"target")
    x=logits[active];y=target[active]
    if not x.numel():return x.sum()
    _finite(x);_finite(y)
    if bool(((y!=0)&(y!=1)).any()):raise ValueError("explicit native zero/one binary class required; no thresholding")
    return F.binary_cross_entropy_with_logits(x,y)


def masked_regression_mse(prediction,target,*,observed,applicable,applicability_known):
    """Equality readouts only, within one declared native unit/assay partition."""
    active=_active(prediction,observed,applicable,applicability_known)
    _numeric(prediction,target,"target")
    x=prediction[active];y=target[active]
    if not x.numel():return x.sum()
    _finite(x);_finite(y)
    return (x-y).square().mean()


def _interval_log_probability(mean,scale,lower,upper):
    if bool((lower>=upper).any()):raise ValueError("source interval requires explicit positive width, never equality imputation")
    span=upper-lower
    width=span/scale
    midpoint=(lower+span*.5-mean)/scale
    narrow=(width*(1+midpoint.abs()))<=1e-3
    result=torch.empty_like(mean)
    if narrow.any():
        # Four-point Gauss-Legendre fallback when log-CDF differences lose
        # precision. The density varies by <=~1e-3 over this narrow regime.
        # Source bounds remain exact; this is declared floating loss arithmetic.
        nodes=mean.new_tensor([-.8611363115940526,-.3399810435848563,.3399810435848563,.8611363115940526])
        weights=mean.new_tensor([.34785484513745385,.6521451548625461,.6521451548625461,.34785484513745385])
        z=midpoint[narrow,None]+width[narrow,None]*.5*nodes
        result[narrow]=(torch.log(span[narrow])-torch.log(scale[narrow])-math.log(2)
            +torch.logsumexp(torch.log(weights)[None,:]-.5*z.square()-.5*_LOG2PI,dim=1))
    regular=~narrow
    if regular.any():
        lo=(lower[regular]-mean[regular])/scale[regular]
        hi=(upper[regular]-mean[regular])/scale[regular]
        # Right-tail survival subtraction avoids cancellation of CDFs near one.
        positive=lo>=0
        a=torch.empty_like(lo);b=torch.empty_like(lo)
        if positive.any():
            a[positive]=torch.special.log_ndtr(-lo[positive])
            b[positive]=torch.special.log_ndtr(-hi[positive])
        if (~positive).any():
            a[~positive]=torch.special.log_ndtr(hi[~positive])
            b[~positive]=torch.special.log_ndtr(lo[~positive])
        if bool((b>=a).any()):raise ValueError("interval probability unresolvable at declared floating precision")
        result[regular]=a+torch.log(-torch.expm1(b-a))
    _finite(result)
    return result


def gaussian_censored_nll(mean,scale,*,relation,value,lower,upper,observed,applicable,applicability_known):
    """Gaussian native-coordinate likelihood, one homogeneous endpoint/assay.

    equal uses a density; left/right use CDF/survival at a source-provided bound;
    interval uses its probability mass. Native < versus <= stays in the data
    contract (identical probabilities under this continuous likelihood). A bare
    censor flag never supplies a bound. No averages/ratios become thresholds.
    """
    active=_active(mean,observed,applicable,applicability_known)
    for x,name in ((scale,"scale"),(value,"value"),(lower,"lower"),(upper,"upper")):_numeric(mean,x,name)
    if type(relation) not in (tuple,list) or len(relation)!=mean.numel():raise ValueError("one explicit native censor kind per row required")
    mu=mean[active];sigma=scale[active]
    if not mu.numel():return mu.sum()+sigma.sum()
    _finite(mu);_finite(sigma)
    if bool((sigma<=0).any()):raise ValueError("positive finite native-unit Gaussian scale required")
    kinds=[relation[i] for i in active.nonzero(as_tuple=False).flatten().tolist()]
    if any(type(k) is not str or k not in ("equal","left","right","interval") for k in kinds):
        raise ValueError("source-supported equality/left/right/interval kind required")
    val=value[active];lo=lower[active];hi=upper[active]
    out=torch.empty_like(mu)
    for kind in ("equal","left","right","interval"):
        selected=torch.tensor([k==kind for k in kinds],device=mu.device,dtype=torch.bool)
        if not selected.any():continue
        m=mu[selected];s=sigma[selected]
        if kind=="equal":
            y=val[selected];_finite(y)
            out[selected]=.5*((y-m)/s).square()+torch.log(s)+.5*_LOG2PI
        elif kind=="left":
            bound=hi[selected];_finite(bound)
            out[selected]=-torch.special.log_ndtr((bound-m)/s)
        elif kind=="right":
            bound=lo[selected];_finite(bound)
            out[selected]=-torch.special.log_ndtr((m-bound)/s)
        else:
            l=lo[selected];u=hi[selected];_finite(l);_finite(u)
            out[selected]=-_interval_log_probability(m,s,l,u)
    _finite(out)
    return out.mean()


def desired_interval_penalty(prediction,lower,upper,*,active):
    """Desired-set distance surrogate, never an observed-label likelihood.

    Bounds use the same declared native coordinate as predictions. Callers keep
    this separate from experimental objectives and do not form a universal score.
    """
    _reference(prediction)
    for x,name in ((lower,"lower"),(upper,"upper")):_numeric(prediction,x,name)
    if (not isinstance(active,torch.Tensor) or active.dtype!=torch.bool
            or active.shape!=prediction.shape or active.device!=prediction.device):
        raise ValueError("explicit desired applicability mask required")
    p=prediction[active];l=lower[active];u=upper[active]
    if not p.numel():return p.sum()
    _finite(p);_finite(l);_finite(u)
    if bool((l>u).any()):raise ValueError("ordered desired bounds required")
    return (F.relu(l-p).square()+F.relu(p-u).square()).mean()


@dataclass(frozen=True)
class AdmetObjectiveContract:
    definition: object
    assay_context: object
    molecule_class: str | None = None

    def __post_init__(self):
        if not isinstance(self.definition,data.EndpointDefinition) or not isinstance(self.assay_context,data.AssayContext):
            raise ValueError("typed endpoint definition and known assay context required")
        if self.molecule_class is not None and self.molecule_class not in data.MOLECULE_CLASSES:
            raise ValueError("explicit source chemistry/polymer class required")
    @property
    def unit(self):return self.definition.unit
    @property
    def partition_key(self):
        return data.digest(data.canonical(dict(endpoint_definition=self.definition.binding(),
            native_context=self.assay_context.as_mapping(),unit=self.unit,unit_basis=self.definition.unit_basis,
            molecule_class=self.molecule_class)))
    def observed_loss(self,prediction,observation,*,source_admission=None):
        """Actual-source entry: never trust generic gate dictionaries as admission.

        No positive experimental source adapter is currently integrated. Empty,
        unknown/inapplicable or nonexperimental targets contribute differentiable
        zero; a usable measured target must pass the verified native task reader.
        """
        _reference(prediction)
        if not isinstance(observation,data.AdmetObservation) or prediction.numel()!=1:
            raise ValueError("one typed native observation/prediction required")
        if (data.canonical(observation.native()["endpoint"])!=self.definition.native_bytes
                or observation.context.native_bytes!=self.assay_context.native_bytes):
            raise ValueError("source endpoint/native-unit/assay partition mismatch")
        if self.molecule_class is None or self.molecule_class!=observation.entity.molecule_class:
            raise ValueError("actual supervision requires explicit matching molecule class")
        if not observation.experimental_measurement_candidate:
            return prediction[:0].sum()
        # This gate intentionally rejects until a real admitted task adapter is
        # reviewed/integrated; no Decimal-to-float label coercion happens here.
        observation.require_source_admission(proof=source_admission)
        raise ValueError("verified ADMET numeric task adapter not yet integrated")

    def _check(self,kind,native_unit):
        if self.definition.objective_kind!=kind:raise ValueError("explicit endpoint-specific objective required")
        if (kind=="binary" and self.definition.unit_applicability=="not_applicable"
                and self.unit is None and native_unit is None):return
        if (self.definition.unit_applicability!="declared_native_unit" or self.definition.unit_basis=="unresolved"
                or self.unit is None or native_unit!=self.unit or type(native_unit) is not str):
            raise ValueError("exact declared native unit required; no conversion or automatic normalization")
    def point_loss(self,prediction,target,*,native_unit,**masks):
        self._check("regression",native_unit)
        return masked_regression_mse(prediction,target,**masks)
    def binary_loss(self,logits,target,*,native_unit,**masks):
        self._check("binary",native_unit)
        return masked_binary_bce(logits,target,**masks)
    def censored_loss(self,mean,scale,*,native_unit,**targets_and_masks):
        self._check("regression",native_unit)
        return gaussian_censored_nll(mean,scale,**targets_and_masks)


def require_single_admet_partition(contracts):
    contracts=tuple(contracts)
    if (not contracts or any(not isinstance(c,AdmetObjectiveContract) for c in contracts)
            or len({c.partition_key for c in contracts})!=1):
        raise ValueError("incomparable assay/readout/species/matrix/native-unit partitions cannot be pooled")
    return contracts[0]


def require_observational_admet_admission(proof=None):
    """Fail closed: no verified native ADMET task adapter is currently integrated.

    This is an integration gate, not a boolean proof API. Source-native candidate
    projections, hashes, data schemas, desired requests, and all-True dictionaries
    never establish experimental model-supervision eligibility.
    """
    raise ValueError("verified source-specific ADMET task reader/admission required before observational model supervision")
