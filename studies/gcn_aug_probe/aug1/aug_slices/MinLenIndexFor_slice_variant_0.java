/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  void test(@IndexFor("this.arrayLen2") int i) {
    @Positive
    int j = arrayLen2[i];
    @Positive
    int j2 = arrayLen2[1];
    @Positive
  }
