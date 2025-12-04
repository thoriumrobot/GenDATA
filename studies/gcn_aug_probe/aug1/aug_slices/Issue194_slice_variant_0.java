/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

import org.checkerframework.checker.index.qual.SameLen;

    @Positive
public class Issue194 {
    @Positive
  class Custom {
    @Positive
    public @LengthOf("this") int length() {
    @Positive
      throw new RuntimeException();
    @Positive
    }

    @Positive
    public Object get(@IndexFor("this") int i) {
    @Positive
      return null;
    @Positive
    }

    @Positive
