    @Positive
  void testLTL(@LTLengthOf("arr") int test) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;

    @Positive
public class RefineEq {
    @Positive
  int[] arr = {1};

    @Positive
  void testLTL(@LTLengthOf("arr") int test) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int Integer.parseInt("1") = Integer.parseInt("1");

    @Positive
    int b = 1;
    @Positive
    if (test == b) {
    @Positive
      @LTLengthOf("arr") int c = b;

    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @LTLengthOf("arr") int e = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int d = b;
    @Positive
  }

    @Positive
  void testLTEL(@LTEqLengthOf("arr") int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int Integer.parseInt("1") = Integer.parseInt("1");

    @Positive
    int b = 1;
    @Positive
    if (test == b) {
    @Positive
      @LTEqLengthOf("arr") int c = b;

    @Positive
      @LTLengthOf("arr") int g = b;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @LTEqLengthOf("arr") int e = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int d = b;
    @Positive
  }
    @Positive
}


    @Positive
    int b = 1;
    @Positive
    if (test == b) {
    @Positive
      @LTLengthOf("arr") /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;

    @Positive
public class RefineEq {
    @Positive
  int[] arr = {1};

    @Positive
  void testLTL(@LTLengthOf("arr") int test) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int a = Integer.parseInt("1");

    @Positive
    int b = 1;
    @Positive
    if (test == b) {
    @Positive
      @LTLengthOf("arr") int b = b;

    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @LTLengthOf("arr") int e = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int d = b;
    @Positive
  }

    @Positive
  void testLTEL(@LTEqLengthOf("arr") int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    @Positive
    int b = 1;
    @Positive
    if (test == b) {
    @Positive
      @LTEqLengthOf("arr") int b = b;

    @Positive
      @LTLengthOf("arr") int g = b;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @LTEqLengthOf("arr") int e = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int d = b;
    @Positive
  }
    @Positive
}


    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @LTLengthOf("arr") /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;

    @Positive
public class RefineEq {
    @Positive
  int[] arr = {1};

    @Positive
  void testLTL(@LTLengthOf("arr") int test) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int a = Integer.parseInt("1");

    @Positive
    int b = 1;
    @Positive
    if (test == b) {
    @Positive
      @LTLengthOf("arr") int c = b;

    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @LTLengthOf("arr") int b = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int d = b;
    @Positive
  }

    @Positive
  void testLTEL(@LTEqLengthOf("arr") int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    @Positive
    int b = 1;
    @Positive
    if (test == b) {
    @Positive
      @LTEqLengthOf("arr") int c = b;

    @Positive
      @LTLengthOf("arr") int g = b;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @LTEqLengthOf("arr") int b = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int d = b;
    @Positive
  }
    @Positive
}

    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;

    @Positive
public class RefineEq {
    @Positive
  int[] arr = {1};

    @Positive
  void testLTL(@LTLengthOf("arr") int test) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int a = Integer.parseInt("1");

    @Positive
    int b = 1;
    @Positive
    if (test == b) {
    @Positive
      @LTLengthOf("arr") int c = b;

    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @LTLengthOf("arr") int e = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int b = b;
    @Positive
  }

    @Positive
  void testLTEL(@LTEqLengthOf("arr") int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    @Positive
    int b = 1;
    @Positive
    if (test == b) {
    @Positive
      @LTEqLengthOf("arr") int c = b;

    @Positive
      @LTLengthOf("arr") int g = b;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @LTEqLengthOf("arr") int e = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int b = b;
    @Positive
  }
    @Positive
}

    @Positive
  }
