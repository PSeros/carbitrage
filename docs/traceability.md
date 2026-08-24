# Traceability

Every number reports where it comes from, and the components sum:

```python
print(result.breakdown_markdown("Buy the EV now"))
```

```
| Purchase price of Hyundai Inster | -23,900.00 |
| Setup cost for Hyundai Inster    |  -1,500.00 |
| Residual value of Hyundai Inster |   7,548.98 |
| Energy for Hyundai Inster        |  -4,313.10 |
| Insurance for Hyundai Inster     |  -4,118.46 |
| Maintenance and wear             |  -1,647.38 |
| Disposal of LPG incumbent        |   1,500.00 |
| Purchase premium, paid month 4   |   3,960.78 |
| Greenhouse-gas quota credit      |   1,625.16 |
| **Total**                        | **-20,844.03** |
```

If you cannot ask "where does this €20,844 come from" and get an answer that
adds up, the library has failed at its actual job — which is not arithmetic but
justification. `breakdown()` asserts the sum on every call.

To compare where the money goes across alternatives rather than within one,
`breakdown_frame()` puts the components down the rows and the alternatives
across the columns, each column summing to that alternative's NPV. A component
one alternative never uses reads as a zero, because that is what it is:

```python
result.breakdown_frame(decimals=0)
```

```
             Buy the EV now  Lease the EV  Repair now, replace in 2 years
component
ACQUISITION          -25400         -1500                          -27123
LEASE                     0        -17387                               0
ENERGY                -4313         -4313                           -5102
INSURANCE             -4118         -4118                           -3731
MAINTENANCE           -1647         -1647                           -2811
TAX                       0             0                            -310
SUBSIDY                5586          5586                            4785
RESIDUAL               7549             0                           11518
DISPOSAL               1500          1500                               0
```

It needs pandas, so install the `frames` extra. `decimals` is the only rounding
in it; omit it and you get full precision.

## The same breakdown, under uncertainty

`result.breakdown_frame()` is deterministic: every figure in it is the base
case's. `MonteCarlo.breakdown_frame()` is the same table read off a simulation,
each cell a statistic over the trials rather than one number. The rows are the
same components, the columns gain a second level, and the statistics run low to
high with the mean where the median would sit:

```python
simulation = monte_carlo(
    case,
    {"home_electricity_price": Triangular(0.22, 0.30, 0.45),
     "residual_rate": Normal(0.15, 0.025)},
    n=1_000,
    seed=20260821,
)
simulation.breakdown_frame(decimals=0)
```

```
alternative Buy the EV now               Lease the EV               Repair now, replace in 2 years
statistic               p5   mean    p95           p5   mean    p95           p5    mean     p95
component
ACQUISITION         -25400 -25400 -25400        -1500  -1500  -1500       -27123  -27123  -27123
LEASE                    0      0      0       -17387 -17387 -17387            0       0       0
ENERGY               -5393  -4538  -3791        -5393  -4538  -3791        -5815   -5251   -4758
INSURANCE            -4118  -4118  -4118        -4118  -4118  -4118        -3731   -3731   -3731
MAINTENANCE          -1647  -1647  -1647        -1647  -1647  -1647        -2811   -2811   -2811
TAX                      0      0      0            0      0      0         -310    -310    -310
SUBSIDY               5586   5586   5586         5586   5586   5586         4785    4785    4785
RESIDUAL              5684   7669   9982            0      0      0         9663   11594   13722
DISPOSAL              1500   1500   1500         1500   1500   1500            0       0       0
NPV                 -23102 -20949 -18428       -22960 -22105 -21358       -24894  -22847  -20591
```

A component nothing sampled reaches shows the same figure in all three columns.
That is a fact about the run, not a rounding artefact: no draw moved the
purchase price, so `ACQUISITION` has no spread to report.

**Only the mean row adds up.** Expectation is linear, so the component means
sum to the `NPV` mean exactly. No percentile does, and the gap is the point:
the component fifth percentiles above sum to -23,788 for the EV, but its `NPV`
fifth percentile is -23,102. Totalling the fifth percentiles describes a trial
in which electricity ran expensive *and* the residual collapsed at the same
time, and the simulation drew that trial rarely. The `NPV` row is taken from
the paired totals, so it is the narrower figure and the one to quote.

`expected_breakdown()` and `component_percentiles(name)` are the same readings
without pandas, and `component_npv(name, Component.ENERGY)` is one component's
raw column, paired row by row with every other column of the simulation.
