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
