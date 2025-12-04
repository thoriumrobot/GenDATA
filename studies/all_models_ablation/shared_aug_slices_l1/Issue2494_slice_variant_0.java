/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
// Applied transformations: attempted_ternary_operator, attempted_logical_expression

    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public final class Issue2494 {

    @Positive
  static final long @MinLen(1) [] factorials = {
    @Positive
    1L,
    @Positive
    1L,
    @Positive
    1L * 2,
    @Positive
    1L * 2 * 3,
    @Positive
    1L * 2 * 3 * 4,
    @Positive
    1L * 2 * 3 * 4 * 5,
    @Positive
