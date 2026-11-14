# Native causal forest-token windows (unadmitted implementation)

The native forest token decoder previously projected each output position independently of its preceding output tokens. This source supplement adds an **isolated** trainable next-token recurrent operator for a *single free forest node*. No native vocabulary, frozen source codebook, flow field, forest identity, or production training gate is replaced.

For a source-bound native field context \(h_0\), observed token identifiers \(y_0,\ldots,y_{T-1}\), frozen source codebook \(E\), an input projection \(P\), GRU \(G\), and existing decoder projection \(D\), the teacher-forced recurrence is

\[
 u_t = \begin{cases} 0 &t=0,\\ P E[y_{t-1}]&t>0,\end{cases}\qquad
 h_t=G(u_t,h_{t-1}),\qquad
 q_t=D\!\left(\operatorname{SiLU}(h_t+s_{t\bmod W})\right).
\]

The implementation applies the existing native μP-aware readout multiplier and the original masked nearest-codebook score to \(q_t\). Here \(W\leq4096\) is a declared finite window capacity. The full hidden state carries across window boundaries; no future token \(y_t\) or subsequent token is used to calculate the score for position \(t\). The stop control appears exactly once at the final supervised position. The teacher loss is the average exact next-token negative log likelihood, computed in bounded score chunks. Greedy inference feeds back its *own* previous predictions and returns either a final-stop result or an explicit budget-exhausted incomplete result. An incomplete result is never serialized as completed content.

The implementation validates finite total-token/window/score budgets before token scoring, source-codebook identity and exact native decoder configuration, and strict shape/source-matched checkpoint loads. CPU reference tests verify causality across boundaries, next-token teacher/inference equality, loss and finite gradients, immutable axes, final-stop policy, explicit shortfall, configuration drift refusal, and checkpoint round-trip with no forward RNG draws. Tests do **not** consume optimizer updates or campaign slots.

**Unresolved qualification:** This is a stand-alone software operator; it is not registered in the complete native owner/factory/μP parameter-role ledger. The native field context still uses the existing full declared skeleton; this prototype does not prove exclusion of future or sibling *structural* information or source-supervision rights/splits. It does not implement batch cursor/RNG/optimizer recovery for a counted run. Long-output representation and true long-output *trained capability* are separate claims. No real original-training source was admitted, no optimizer was stepped, and neither the 24 μP proxies nor either target run has started because of this change. The hybrid teacher, raw-cell/BETSE, SCRI, topology and directed evidence lanes remain unchanged.
