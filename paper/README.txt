NormFlowSDE manuscript
======================

Title: NormFlowSDE: Normalizing Flows for Amortized Bayesian State Estimation
Focus: model-free amortized inference from paired state/observation trajectories.
Motivation: extend BayesFlow-style conditional parameter inference to evolving
states by combining normalizing flows with neural SDE-style temporal updates.
The introduction contrasts variational and adversarial neural SDE formulations,
motivates testing flow-transformed noise for state uncertainty, and presents
the weak-convergence foundation and NSF/FlowSDE robot benchmarks.
Model-free means no hand-specified physical transition equations are required;
representative training data and an initial-state treatment are still needed.
Known physics can optionally be supplied. Sim-to-real mismatch motivates learning
from deployment data; the saved experiments do not establish transfer robustness.
Mean recentering belongs to the supporting diffusion-limit analysis and is not
required for finite-step likelihood evaluation or conditional state sampling.
The methods section derives every term of the proposed composite training loss
and gives a short diffusion-limit proof sketch. The weak-convergence assumptions,
theorem, and full proof are in sections/Appendix.tex.
Experiments: Lorenz-96 partial-observation inference, REASSEMBLE, then ALOHA.

Main source: main.tex. Output: main.pdf after compilation.
Fresh portrait review export: normflowsde-state-estimation.pdf.
Section 3 introduces the observation-conditioned transition, derives Gaussian
reconstruction, self-sampled transition entropy, boundary consistency, and the
full-horizon energy score, and specifies multiple shooting and optimization.
It distinguishes this proposed composite loss and five-update rollout schedule
from the current reference-transition NLL/ensemble-score implementation. No
claim is made that the proposed loss produced the archived experimental rows.
The main text includes a short diffusion-limit proof sketch; Appendix A retains
the complete theorem and proof for the centered inference scheme. Each context encodes a trailing
observation window ending before the predicted state; the supplied initial state
initializes the sampled process. This formulation differs from the archived
full-sequence smoothing implementations. The frozen ALOHA particle-filter and REASSEMBLE Gaussian-
smoother evaluations do not establish standalone FlowSDE and NSF performance.
Experiments follow in Section 4.
Format: official, unmodified TMLR style in anonymous review mode.
Status: complete research-paper draft with a retrospective empirical evaluation;
not a claim of independently rerun or submission-ready experiments.

Build with a TeX installation:
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
or:
  tectonic main.tex
After a successful build, refresh the separately named review copy with:
  cp main.pdf normflowsde-state-estimation.pdf

Regenerate tables and figures (Python 3 + Matplotlib):
  python scripts/build_artifacts.py
This uses frozen local evidence and does not train any model.
Use --refresh only to deliberately update evidence from the research repository.
Figure 1 is a vector schematic of the complete estimator. Its illustrative curves
are not experimental data. scripts/draw_model_overview.py produces PDF and SVG.

Check citations, cross-references, input paths, and evidence structure in the
active input/include tree rooted at main.tex (including generated tables):
  python scripts/validate_manuscript.py
Within the research checkout, also verify original artifact hashes:
  python scripts/validate_manuscript.py --check-sources

Contents:
  sections/       Narrative, method, theorem statements, proofs, experiments
  references.bib  Verified bibliography for the cited work
  evidence/       Saved numerical summaries and original-file SHA-256 hashes
  tables/         Generated LaTeX tables
  figures/        Generated vector PDF figures
  scripts/        Artifact generation and source/evidence validation
  vendor/         Original TMLR example, readme, and license
  EDITORIAL_NOTES.txt  Private provenance, corrections, and remaining work

sections/Appendix.tex is the active mathematical appendix, included where the
former sections/proofs.tex input appeared. sections/proofs.tex is retained as
an inactive historical source; its removed-section references are not compiled.

The PDF uses anonymous author formatting. Author names, affiliations, funding,
and submission-specific declarations should be supplied for the final version.
Do not upload this entire development directory as an anonymous supplement:
the editorial notes and source manifests are private provenance records.

The repository's existing private/not-approved redistribution status is retained.
No publication, submission, repository push, or dataset redistribution was done.
