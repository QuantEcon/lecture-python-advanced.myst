---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(additive_functionals)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Additive and Multiplicative Functionals

```{index} single: Models; Additive functionals
```

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install --upgrade quantecon
```

```{note}
This lecture uses JAX for GPU acceleration. JAX automatically detects and uses available GPUs, falling back to CPU otherwise.

For local GPU support, see the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for CUDA and cuDNN setup.
```
```

---

## Important Note for Complete Review

This review is based on an incomplete lecture fragment (only 35 lines of header/setup content). A comprehensive style guide review requires the full lecture content to properly assess:

### JAX-Specific Rules (qe-jax-008, qe-jax-010)
- Loop patterns and JAX constructs usage
- JAX transformations (jit, vmap, grad)
- Random key handling
- Array operations and functional patterns

### Reference Rules (qe-ref-003, qe-ref-004)
- Internal lecture links
- Cross-series references
- Citation formatting

### Writing Rules Throughout Content
- Mathematical notation (qe-writing-007)
- Logical flow (qe-writing-004)
- Simplicity preference (qe-writing-005)
- Overall clarity and brevity

### Additional Elements Not Present in Fragment
- Code style compliance
- Exercise formatting
- Figure and table formatting
- Mathematical equation formatting
- Index entries throughout content

**Please provide the complete lecture content for a thorough, comprehensive style guide review.**