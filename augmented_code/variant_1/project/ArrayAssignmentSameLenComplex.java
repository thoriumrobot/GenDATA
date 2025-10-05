/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;

// @skip-test until #127 is resolved.

    @Positive
public class ArrayAssignmentSameLenComplex {

    @Positive
  static class Partial {
    @Positive
    private final int[] iValues;

    @Positive
    Partial(@NonNegative int n) {
    @Positive
      iValues = new int[n];
    @Positive
    }
    @Positive
  }

    @Positive
  private final Partial iBase;
    @Positive
  private final @IndexFor("iBase.iValues") int iFieldIndex;

    @Positive
  ArrayAssignmentSameLenComplex(Partial partial, @IndexFor("#1.iValues") int fieldIndex) {
    @Positive
    iBase = partial;
    @Positive
    iFieldIndex = fieldIndex;
    @Positive
  }
    @Positive
}
