// Test case for kelloggm 214
// https://github.com/kelloggm/checker-framework/issues/214

    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;

    @Positive
public class ShiftRight {
    @Positive
  void indexFor(Object[] a, @IndexFor("#1") int i) {
    @Positive
    @IndexFor("a") int o = i >> 2;
    @Positive
    @IndexFor("a") int p = i >>> 2;
    @Positive
  }

    @Positive
  void indexOrHigh(Object[] a, @IndexOrHigh("#1") int i) {
    @Positive
    @IndexOrHigh("a") int o = i >> 2;
    @Positive
    @IndexOrHigh("a") int p = i >>> 2;
    // Not true if a.length == 0
    // :: error: (assignment)
    @Positive
    @IndexFor("a") int q = i >> 2;
    @Positive
  }

    @Positive
  void negative(Object[] a, @LTLengthOf(value = "#1", offset = "100") int i) {
    // Not true for some negative i
    // :: error: (assignment)
    @Positive
    @LTLengthOf(value = "#1", offset = "100") int q = i >> 2;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
