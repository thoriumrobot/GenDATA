    @Positive
import org.checkerframework.checker.index.qual.HasSubsequence;
    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;

    @Positive
public class OnlyCheckSubsequenceWhenAssigningToArray {
    @Positive
  int[] array;

    @Positive
  final @IndexFor("array") int start;

    @Positive
  final @IndexOrHigh("array") int end;

    @Positive
  private OnlyCheckSubsequenceWhenAssigningToArray(
    @Positive
      @IndexFor("array") int s, @IndexOrHigh("array") int e) {
    @Positive
    start = s;
    @Positive
    end = e;
    @Positive
  }

    @Positive
  void testAssignmentToArrayElement(@IndexFor("this") int x, int y) {
    @Positive
    array[start + x] = y;
    @Positive
  }

    @Positive
  void testAssignmentToArray(int[] a) {
    // :: error: (to.not.ltel) :: error: (from.gt.to)
    @Positive
    array = a;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
