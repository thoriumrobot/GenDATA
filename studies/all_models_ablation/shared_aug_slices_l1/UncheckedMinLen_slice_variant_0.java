/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  void addToNonNegative(@NonNegative int l, Object v) {
    // :: error: (assignment)
    @Positive
    Object @MinLen(100) [] o = new Object[l + 1];
    @Positive
    o[99] = v;
    @Positive
  }

    @Positive
  void addToPositive(@Positive int l, Object v) {
    // :: error: (assignment)
    @Positive
    Object @MinLen(100) [] o = new Object[l + 1];
    @Positive
    o[99] = v;
    @Positive
  }

    @Positive
  void addToUnboundedIntRange(@IntRange(from = 0) int l, Object v) {
    // :: error: (assignment)
    @Positive
    Object @MinLen(100) [] o = new Object[l + 1];
    @Positive
    o[99] = v;
    @Positive
  }
