    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;

    @Positive
public class RefineLTE {
    @Positive
  int[] arr = {1};

    @Positive
  void testLTL(@LTLengthOf("arr") int test) {
    // The reason for the parsing is so that the Value Checker
    // can't figure it out but normal humans can.

    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int a3 = Integer.parseInt("3");

    @Positive
    int b = 2;
    @Positive
    if (b <= test) {
    @Positive
      @LTLengthOf("arr") int c = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int c1 = b;

    @Positive
    if (b <= a) {
    @Positive
      int potato = 7;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @LTLengthOf("arr") int d = b;
    @Positive
    }
    @Positive
  }

    @Positive
  void testLTEL(@LTEqLengthOf("arr") int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int a3 = Integer.parseInt("3");

    @Positive
    int b = 2;
    @Positive
    if (b <= test) {
    @Positive
      @LTEqLengthOf("arr") int c = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int c1 = b;

    @Positive
    if (b <= a) {
    @Positive
      int potato = 7;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @LTLengthOf("arr") int d = b;
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
