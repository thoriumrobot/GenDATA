    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;

    @Positive
public final class Loops {
    @Positive
  public static boolean flag = false;

    @Positive
  public void test1a(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
    @Positive
    while (flag) {
      // :: error: (unary.increment)
    @Positive
      offset++;
    @Positive
    }
    @Positive
  }

    @Positive
  public void test1b(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
    @Positive
    while (flag) {
      // :: error: (compound.assignment)
    @Positive
      offset += 1;
    @Positive
    }
    @Positive
  }

    @Positive
  public void test1c(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
    @Positive
    while (flag) {
      // :: error: (compound.assignment)
    @Positive
      offset2 += offset;
    @Positive
    }
    @Positive
  }

    @Positive
  public void test2(int[] a, int[] array) {
    @Positive
    int offset = array.length - 1;
    @Positive
    int offset2 = array.length - 1;

    @Positive
    while (flag) {
    @Positive
      offset++;
    @Positive
      offset2 += offset;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("array") int x = offset;
    // :: error: (assignment)
    @Positive
    @LTLengthOf("array") int y = offset2;
    @Positive
  }

    @Positive
  public void test3(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
    @Positive
    while (flag) {
    @Positive
      offset--;
      // :: error: (compound.assignment)
    @Positive
      offset2 -= offset;
    @Positive
    }
    @Positive
  }

    @Positive
  public void test4(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
    @Positive
    while (flag) {
      // :: error: (unary.increment)
    @Positive
      offset++;
      // :: error: (compound.assignment)
    @Positive
      offset += 1;
      // :: error: (compound.assignment)
    @Positive
      offset2 += offset;
    @Positive
    }
    @Positive
  }

    @Positive
  public void test4(int[] src) {
    @Positive
    int patternLength = src.length;
    @Positive
    int[] optoSft = new int[patternLength];
    @Positive
    for (int i = patternLength; i > 0; i--) {}
    @Positive
  }

    @Positive
  public void test5(
    @Positive
      int[] a,
    @Positive
      @LTLengthOf(value = "#1", offset = "-1000") int offset,
    @Positive
      @LTLengthOf("#1") int offset2) {
    @Positive
    int otherOffset = offset;
    @Positive
    while (flag) {
    @Positive
      otherOffset += 1;
      // :: error: (unary.increment)
    @Positive
      offset++;
      // :: error: (compound.assignment)
    @Positive
      offset += 1;
      // :: error: (compound.assignment)
    @Positive
      offset2 += offset;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf(value = "#1", offset = "-1000") int x = otherOffset;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
