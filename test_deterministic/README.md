Deterministic augmentation tests

This folder contains small Java snippets used to assert that the JDT-based augmentation produces visible textual differences (beyond headers).

Snippets:
- Demo.java: for-loop summation and simple if/else returning different strings.

Expected: At least one of the prioritized transforms (variable_operation, ternary_operator, mathematical_expression) yields a body change under `--require-change`.


