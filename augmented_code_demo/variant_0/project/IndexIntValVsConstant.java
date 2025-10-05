    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LessThan;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.common.value.qual.ArrayLen;
    @Positive
import org.checkerframework.common.value.qual.IntVal;

    @Positive
public class IndexIntValVsConstant {

    @Positive
  void m() {

    @Positive
    int @ArrayLen(7) [] a1 = new int[] {1, 2, 3, 4, 5, 6, 7};

    @Positive
    @IntVal(2) int i = 2;
    @Positive
    @IntVal(4) int j = 4;

    @Positive
    int[] s0 = internSubsequence(a1, 2, 4);
    @Positive
    int[] s1 = internSubsequence(a1, i, j);
    @Positive
  }

    @Positive
  int @Interned [] internSubsequence(
    @Positive
      int @Interned [] seq,
    @Positive
      @IndexFor("#1") @LessThan("#3") int start,
    @Positive
      @NonNegative @LTLengthOf(value = "#1", offset = "#2 - 1") int end) {
    // dummy implementation
    @Positive
    return new int[0];
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
