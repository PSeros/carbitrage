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

