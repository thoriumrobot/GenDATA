    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class ArrayAssignmentSameLen {

    @Positive
  private final int[] i_array;
    @Positive
  private final @IndexFor("i_array") int i_index;

    @Positive
  ArrayAssignmentSameLen(int[] array, @IndexFor("#1") int index) {
    @Positive
    i_array = array;
    @Positive
    i_index = index;
    @Positive
  }

    @Positive
  void test1(int[] a, int[] b, @LTEqLengthOf("#1") int index) {
    @Positive
    int[] array = a;
    @Positive
        value = {"array", "b"},
    @Positive
        offset = {"0", "-3"})
    // :: error: (assignment)
    @Positive
    int i = index;
    @Positive
  }

    @Positive
  void test2(int[] a, int[] b, @LTLengthOf("#1") int i) {
    @Positive
    int[] c = a;
    // :: error: (assignment)
    @Positive
    @LTLengthOf(value = {"c", "b"}) int x = i;
    @Positive
    @LTLengthOf("c") int y = i;
    @Positive
  }

    @Positive
  void test3(int[] a, @LTLengthOf("#1") int i, @NonNegative int x) {
    @Positive
    int[] c1 = a;
    // See useTest3 for an example of why this assignment should fail.
    @Positive
        value = {"c1", "c1"},
    @Positive
        offset = {"0", "x"})
    // :: error: (assignment)
    @Positive
    int z = i;
    @Positive
  }

    @Positive
  void test4(
    @Positive
      int[] a,
    @Positive
              value = {"#1", "#1"},
    @Positive
              offset = {"0", "#3"})
    @Positive
          int i,
    @Positive
      @NonNegative int x) {
    @Positive
    int[] c1 = a;
    @Positive
        value = {"c1", "c1"},
    @Positive
        offset = {"0", "x"})
    @Positive
    int z = i;
    @Positive
  }

    @Positive
  void useTest3() {
    @Positive
    int[] a = {1, 3};
    @Positive
    test3(a, 0, 10);
    @Positive
  }
    @Positive
}
