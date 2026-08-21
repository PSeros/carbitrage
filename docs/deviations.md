# Deviations from the original brief

Two, both deliberate:

1. **Builders take a `Context`**, not the brief's literal
   `flows(vehicle, usage, timeline)`. Those signatures cannot express a
   replacement chain — see the method note above for why the leg's start period
   is an economic quantity rather than an implementation detail.
2. **Excel is an output, not an input.** The brief specified an
   Excel → objects loader. Cases are defined in Python here; `carbitrage.reporting`
   writes a *result* workbook (ranking, breakdown, full cash-flow grid,
   scenarios) as values rather than formulas, so there is never a second
   implementation of the engine to drift out of step.

The real-option lattice for the value of deferring is out of scope for now.

