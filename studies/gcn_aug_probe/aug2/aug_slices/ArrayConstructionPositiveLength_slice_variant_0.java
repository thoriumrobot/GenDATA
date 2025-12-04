/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
import org.checkerframework.common.value.qual.*;

    @Positive
public class ArrayConstructionPositiveLength {

    @Positive
  public void makeArray(@Positive int max_values) {
    @Positive
    String @MinLen(1) [] a = new String[max_values];
    @Positive
  }
    @Positive
}
