/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.HasSubsequence;
    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;

    @Positive
public class BasicSubsequence2 {
    @Positive
  int[] array;

    @Positive
  int[] array2;

    @Positive
  final @IndexFor("array") int start;

    @Positive
  final @IndexOrHigh("array") int end;

    @Positive
  private BasicSubsequence2(@IndexFor("array") int s, @IndexOrHigh("array") int e) {
    @Positive
    start = s;
    @Positive
    end = e;
    @Positive
  }

    @Positive
  void testStartIndex(@IndexFor("this") int x) {
    @Positive
    @IndexFor("array") int y = x + start;
    @Positive
  }

    @Positive
  void testViewpointAdaption(@IndexFor("this") int x) {
    @Positive
    @IndexFor("array2") int y = x + start;
    @Positive
  }

    @Positive
  void testArrayAccess(@IndexFor("this") int x) {
    @Positive
    int y = array[x + start];
    @Positive
  }

    @Positive
  void testCommutativity(@IndexFor("this") int x) {
    @Positive
    @IndexFor("array") int y = start + x;
    @Positive
  }
    @Positive
}
