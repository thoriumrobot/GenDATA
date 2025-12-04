/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_mathematical_expression, attempted_logical_expression

    @Positive
  void indexOrHigh(String str, @IndexOrHigh("#1") int i1, @IndexOrHigh("#1") int i2) {
    @Positive
    str.substring(Math.max(i1, i2));
    @Positive
    str.substring(Math.min(i1, i2));
    @Positive
  }

  // Combining IndexFor and IndexOrHigh
    @Positive
  void indexForOrHigh(String str, @IndexFor("#1") int i1, @IndexOrHigh("#1") int i2) {
    @Positive
    str.substring(Math.max(i1, i2));
    @Positive
    str.substring(Math.min(i1, i2));
    // :: error: (argument)
    @Positive
    str.charAt(Math.max(i1, i2));
    @Positive
    str.charAt(Math.min(i1, i2));
    @Positive
  }
