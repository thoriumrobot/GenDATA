/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  int @PolySameLen [] samelen_identity(int @PolySameLen [] a) {
    @Positive
    int @SameLen("a") [] x = a;
    @Positive
    return a;
    @Positive
  }

    @Positive
  @PolyUpperBound int ubc_identity(@PolyUpperBound int a) {
    @Positive
    return a;
    @Positive
  }

  // SameLen tests
    @Positive
  void samelen_id(int @SameLen("#2") [] a, int[] a2) {
    @Positive
    int[] banana;
    @Positive
    int @SameLen("a2") [] b = samelen_identity(a);
    // :: error: (assignment)
    @Positive
    int @SameLen("banana") [] c = samelen_identity(b);
    @Positive
  }
