# What it can tell you

```python
result.best()                       # the winning Evaluation
result.ranking()                    # every alternative, best first
result.is_material()                # is the lead bigger than 3 % of PV, or noise?
result.breakdown("Buy the EV now")  # labelled components that sum to the total
result.incremental(a, b)            # differential stream, its PV, IRR and payback
result.to_markdown(); result.to_frame()
result.breakdown_frame(decimals=0)  # components x alternatives, side by side
```

**Naming the parameter** — a study has to say *which* number it is varying.
Mark the number where you write it and it names itself; otherwise use one of
the short aliases, or the dotted path:

```python
repair = Uncertain(1_700, "repair_bill")          # a float that carries a name
keep = Alternative(renault, Purchase(upfront_extra=repair), label="Renault")
...
result.one_way(repair, [800, 1_700, 3_000])       # the mark itself
result.one_way("repair_bill", [800, 1_700, 3_000])  # or its label

params.uncertainties(case)   # every mark in the case, and where it sits
params.find(case, "life")    # locate what nobody marked
params.ALIASES               # the short names every case shares
```

A mark is an ordinary float, so the base case runs on it untouched. It shadows
an alias of the same name, and one label may mark several fields, which is how
you say *these move together*.

**Sensitivity** — every one of these re-runs the full monthly cash-flow engine:

```python
result.switch_point("lpg_price", (a, b))       # solved with brentq, None if no crossing
result.switch_point_report("lpg_price", (a, b)) # ... and why, when there is none
result.one_way("annual_km", [5_000, 12_000, 30_000])
result.two_way("annual_km", rows, "lpg_price", cols)
result.tornado({"annual_km": Range(5_000, 30_000),
                "residual_rate": Range(0.7, 1.4, relative=True)})
result.monte_carlo({"lpg_price": Triangular(0.80, 0.99, 1.40)},
                   between=(a, b), correlation=[[1, .8], [.8, 1]], n=5_000)
```

`monte_carlo` returns the distribution of the **difference**, not of each
alternative separately, and reports `P(a beats b)`. Two alternatives that share
an energy price move together; the spread of each one alone says nothing about
how often one wins.

**Scenarios** — expected value *and* minimax regret, because when probabilities
are soft the two rules can disagree, and that disagreement is the finding:

```python
ScenarioSet([
    Scenario("subsidy expires", {"...incentives[0].available": 0}, probability=0.3),
    Scenario("incumbent dies early", {"...upfront_extra": 6_000}, probability=0.2),
]).run(case).to_markdown()
```

**Optimal replacement age** — the textbook answer to "how long should I keep
it", by minimising EAC over candidate holding periods:

```python
optimal_replacement_age(alt, timeline=tl, candidates=[3, 4, 6, 8, 10], usage=usage)
```

The classical U-shaped curve only appears when running costs actually grow with
age. With flat costs the annuity falls monotonically and the "optimum" is just
the longest candidate — a statement about your inputs, not about replacement
policy.

