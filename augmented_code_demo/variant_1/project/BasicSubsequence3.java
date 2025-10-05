    @Positive
import org.checkerframework.checker.index.qual.HasSubsequence;
    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.LessThan;

    @Positive
public class BasicSubsequence3 {
    @Positive
  int[] array;

    @Positive
  int[] array2;

    @Positive
  final @IndexFor("array") int start;

    @Positive
  final @IndexOrHigh("array") int end;

    @Positive
  private BasicSubsequence3(@IndexFor("array") int s, @IndexOrHigh("array") int e) {
    @Positive
    start = s;
    @Positive
    end = e;
    @Positive
  }

    @Positive
  void testStartIndex(@IndexFor("array") @LessThan("this.end") int x) {
    @Positive
    @IndexFor("this") int y = x - start;
    @Positive
  }

    @Positive
  void testViewpointAdaption(@IndexFor("array2") @LessThan("end") int x) {
    @Positive
    @IndexFor("this") int y = x - start;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
