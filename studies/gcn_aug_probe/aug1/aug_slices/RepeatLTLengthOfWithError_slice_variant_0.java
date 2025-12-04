/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
  public void withpostconditionsfunc1() {
    @Positive
    v1 = value1.length() - 3; // condition not satisfied here
    @Positive
    v2 = value2.length() - 3;
    @Positive
    v3 = value3.length() - 3;
    @Positive
  }

    @Positive
  public boolean withcondpostconditionsfunc2() {
    @Positive
    v1 = value1.length() - 3; // condition not satisfied here
    @Positive
    v2 = value2.length() - 3;
    @Positive
    v3 = value3.length() - 3;
    // :: error: (contracts.conditional.postcondition)
    @Positive
    return true;
    @Positive
  }
