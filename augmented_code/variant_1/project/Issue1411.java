/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// Test case for Issue 1411:
// https://github.com/typetools/checker-framework/issues/1411

    @Positive
import org.checkerframework.dataflow.qual.Pure;

    @Positive
interface IGeneric<V> {
    @Positive
  public V get();
    @Positive
}

    @Positive
interface IConcrete extends IGeneric<char[]> {}

    @Positive
public class Issue1411 {
    @Positive
  static void m(IConcrete ic) {
    @Positive
    char[] val = ic.get();
    @Positive
  }
    @Positive
}
