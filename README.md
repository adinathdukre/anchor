<h1 align="center">
<strong>Spurious Time-Anchor Attribution in Wearable and Molecular Biobanks: Detection and Correction with ANCHOR</strong>
</h1>

<div align="center">

<a href="https://git.io/typing-svg">
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&pause=1000&color=1C5CAB&center=true&width=640&lines=Few+noisy+nights+attenuate+and+mix+anchor+effects.;Leakage+grows+more+certain+as+cohorts+grow.;ANCHOR%3A+diagnose%2C+calibrate%2C+attribute."
alt="Typing SVG"
style="margin-bottom:-10px; display:block;" />
</a>

[![RECOMB 2027](https://img.shields.io/badge/RECOMB_2027-Manuscript-8C1B13?style=for-the-badge)](#-citation)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.9-3776AB?style=for-the-badge&logo=python&logoColor=white)](#️-installation)
[![Dependency](https://img.shields.io/badge/Dependency-NumPy_only-013243?style=for-the-badge&logo=numpy&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Visitors](https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fgithub.com%2Fadinathdukre%2Fanchor&label=Views&countColor=%231c5cab&style=for-the-badge)](https://visitorbadge.io/status?path=https%3A%2F%2Fgithub.com%2Fadinathdukre%2Fanchor)

<h3>🧠 <a href="#-method">Method</a> &nbsp;|&nbsp; ⚡ <a href="#-quick-start">Quick start</a> &nbsp;|&nbsp; 🔁 <a href="#-nhanes-replication">NHANES</a></h3>

**[Adinath Madhavrao Dukre](https://github.com/adinathdukre), Imran Razzak**

Mohamed bin Zayed University of Artificial Intelligence, Abu Dhabi, UAE

<img src="https://raw.githubusercontent.com/genmilab/VGS-Decoding/main/docs/assets/genmilab-logo.png" alt="GenMI Lab" height="60"/>

</div>

## 🔥 News
- **[Nov 2026]** 🚀 Code for ANCHOR and the public NHANES analysis is released.
- 📄 The accompanying paper was prepared for **RECOMB 2027**.

## Overview
Official code for **ANCHOR**, an add-on to the standard analysis of timing exposures in biobanks.

Wearable biobanks summarise a few nights of behaviour as exposures measured from different time anchors, for example the clock time of the last meal *M* and its interval to sleep onset *D = S − M*, and fit both jointly to decide which anchor matters. Averaging a few noisy nights gives an estimator that converges to **Λβ**, where the *attribution matrix* Λ = (Σ<sub>X</sub> + Σ<sub>U</sub> E[1/k])<sup>−1</sup> Σ<sub>X</sub> both attenuates and mixes anchor effects. Because both exposures are built from the same events, night-to-night error and habits have differently shaped covariances, so signal leaks from one anchor to the other, and the false anchor signal becomes more likely, not less, as cohorts grow.

## 📖 Contents
- [🧠 Method](#-method)
- [⛏️ Installation](#️-installation)
- [⚡ Quick start](#-quick-start)
- [🔁 NHANES replication](#-nhanes-replication)
- [🔒 Data and scope](#-data-and-scope)
- [🗂️ Repository Structure](#️-repository-structure)
- [⚠️ Limitations](#️-limitations)
- [📝 Citation](#-citation)
- [📚 Acknowledgments](#-acknowledgments)
- [📨 Contact](#-contact)
- [📜 License](#-license)

## 🧠 Method

- **Outcome-free leakage check.** Leakage from one anchor to the other vanishes only when the within-person and between-person slopes of one anchor on the other coincide. Both slopes come from the nightly table alone, before any outcome model is fitted (`leak_check`).
- **Calibration with unequal nights.** Each person mean is replaced by the best linear prediction of the habit, shrinking both anchors jointly and people with fewer nights more. A reliability floor caps noise amplification when habits are nearly collinear.
- **Dated nights.** If nightly errors are correlated, the within-person semivariogram of dated nights identifies the night-to-night correlation, and each person's night count is replaced by an effective number of nights.
- **Habitual versus acute effects.** For outcomes measured every night (e.g. CGM), the person-mean slope mixes habitual and acute effects, so the standard one-shot correction over-corrects. ANCHOR solves the mixture for the habitual effect and tests whether the two differ.
- **Basis-free contrasts and negative controls.** Interventional contrasts (later meal at the same bedtime, whole schedule shifted, later bedtime at the same meal) do not depend on the chosen anchors, and anchor-aligned outcome windows act as negative controls.
- **Inference and scale.** A participant bootstrap resamples whole people with all their nights. Calibration does not depend on the outcome, so all outcomes are solved together.

## ⛏️ Installation

```bash
git clone https://github.com/adinathdukre/anchor.git
cd anchor
pip install -e .
```

or directly:

```bash
pip install git+https://github.com/adinathdukre/anchor
```

The `anchor` package needs Python ≥ 3.9 and NumPy only. The NHANES analysis additionally uses pandas.

On macOS with NumPy 2.0 you may see `RuntimeWarning`s from `matmul`. They did not change the results in our checks.

## ⚡ Quick start

A runnable example on simulated data, with a true clock effect and no gap effect:

```bash
python examples/quickstart.py
```

The main calls:

```python
import anchor

# pid: participant id per night; W: nightly anchors (n_nights x 2), e.g. [meal clock M, gap D]
anchor.leak_check(pid, W)                              # do my anchors leak? (no outcome needed)

ids, Wbar, k, Su, _ = anchor.person_summaries(pid, W)
beta = anchor.fit_anchor(Wbar, k, Su, y, C)            # one outcome
B = anchor.fit_anchor_many(Wbar, k, Su, Y, C)          # many outcomes (columns of Y) at once
draws = anchor.bootstrap_many(pid, W, Y, C, B=200)     # participant bootstrap, all outcomes
beta_d, phi = anchor.fit_anchor_dated(pid, W, t, y, C) # dated nights (t in days)
```

> [!TIP]
> Run `leak_check` on any study with repeated nights before fitting two anchors jointly. If the within- and between-person slopes differ, report corrected effects and interventional contrasts rather than which coefficient survives, and record night dates.

## 🔁 NHANES replication

The anchor-basis analysis uses public NHANES 2017–2018 data and needs no special access:

```bash
pip install pandas
sh code/nhanes/download.sh          # about 5.5 MB from the CDC, saved to data_external/
python code/nhanes/nhanes_anchor.py # writes results/nhanes_anchor*.csv
```

## 🔒 Data and scope

This repository contains the ANCHOR package and the public NHANES analysis. It contains no participant-level data and no Human Phenotype Project (HPP) results.

- **HPP.** HPP data are available to bona fide researchers through the project's data-access application ([humanphenotypeproject.org/data-access](https://humanphenotypeproject.org/data-access)), in a secure cloud-based environment, after signing a data-access agreement for non-commercial use.
- **Other paper materials.** The simulation code, the scripts run inside the HPP environment and the aggregate HPP results will be added after data-provider approval.
- **NHANES.** The analysis in `code/nhanes` uses openly available data.

## 🗂️ Repository Structure

```text
anchor/
├── code/
│   ├── anchor/           # ANCHOR library: attribution matrix, calibration with reliability floor,
│   │                     # habitual/acute decomposition, dated nights, participant bootstrap, leak_check
│   └── nhanes/           # NHANES 2017–2018 anchor-basis replication
├── examples/             # quickstart.py on simulated data
├── results/              # NHANES outputs
└── pyproject.toml        # package `anchor-attribution`
```

## ⚠️ Limitations

- Calibration assumes additive nightly error independent of habits.
- Covariates are not calibrated.
- Short logging periods may not capture habits over the timescale of slow outcomes.
- Habitual effects are causal only under exchangeability of habits given covariates; acute effects assume no time-varying confounding.

## 📝 Citation

If you find our paper and code useful in your research, please cite:

```bibtex
@misc{dukre2026anchor,
  title  = {Spurious time-anchor attribution in wearable and molecular biobanks: detection and correction with {ANCHOR}},
  author = {Dukre, Adinath Madhavrao and Razzak, Imran},
  note   = {Manuscript},
  year   = {2026},
  url    = {https://github.com/adinathdukre/anchor}
}
```

## 📚 Acknowledgments

We gratefully acknowledge the participants and study staff of the Human Phenotype Project for their contribution to this work, and thank Tal Shor and other members of Pheno.AI for their support. 

## 📨 Contact
For questions or collaboration, please open an [issue](https://github.com/adinathdukre/anchor/issues) or reach out to [Adinath Madhavrao Dukre](https://github.com/adinathdukre).

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

> [!IMPORTANT]
> ANCHOR is intended for research only. Its estimates are not clinical guidance and must not inform any diagnostic or treatment decision.
