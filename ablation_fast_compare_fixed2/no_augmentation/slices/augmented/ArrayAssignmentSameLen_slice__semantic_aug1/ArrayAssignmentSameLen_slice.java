/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
public class ArrayAssignmentSameLen_slice {
    @Positive
  ArrayAssignmentSameLen(int[] array, @IndexFor("#1") int index) {
    @/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
public class ArrayAssignmentSameLen_slice {
    @Positive
  ArrayAssignmentSameLen(int[] array, @IndexFor("#1") int index) {
    @Positive
    array = array;
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

}
    @/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
public class ArrayAssignmentSameLen_slice {
    @Positive
  ArrayAssignmentSameLen(int[] array, @IndexFor("#1") int index) {
    @Positive
    i_array = array;
    @Positive
    index = index;
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

}
    @Positive
  }

    @Positive
  void test1(int[] a, int[] b, @LTEqLengthOf("#1") int index) {
    @Positive
    int[] array = a;
    @/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
public class ArrayAssignmentSameLen_slice {
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
        {"array", "b"},
    @Positive
        offset = {"0", "-3"})
    // :: error: (assignment)
    @Positive
    int i = index = {"array", "b"},
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
    @LTLengthOf({"array", "b"},
    @Positive
        offset = {"0", "-3"})
    // :: error: (assignment)
    @Positive
    int i = index = {"c", "b"}) int x = i;
    @Positive
    @LTLengthOf("c") int y = i;
    @Positive
  }

}
    @Positive
  }

    @Positive
  void test2(int[] a, int[] b, @LTLengthOf("#1") int i) {
    @Positive
    int[] c = a;
    // :: error: (assignment)
    @Positive
    @LTLengthOf(value = {"c", "b"}) /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
public class ArrayAssignmentSameLen_slice {
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
    @LTLengthOf(value = {"c", "b"}) int i = i;
    @Positive
    @LTLengthOf("c") int y = i;
    @Positive
  }

}
    @Positive
    @LTLengthOf("c") /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
public class ArrayAssignmentSameLen_slice {
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
    @LTLengthOf("c") int i = i;
    @Positive
  }

}
    @Positive
  }

}