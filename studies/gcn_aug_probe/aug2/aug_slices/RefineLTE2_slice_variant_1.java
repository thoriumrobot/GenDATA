/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  public void add(int elt) {
    @Positive
    if (num_values == values.length) {
    @Positive
      values = null;
      // :: error: (unary.increment)
    @Positive
      num_values++;
    @Positive
      return;
    @Positive
    }
    @Positive
    values[num_values] = elt;
    @Positive
    num_values++;
    @Positive
  }
