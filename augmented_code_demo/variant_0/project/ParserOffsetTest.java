    @Positive
import org.checkerframework.checker.index.qual.*;
    @Positive
import org.checkerframework.common.value.qual.*;

    @Positive
public class ParserOffsetTest {

    @Positive
  public void subtraction1(String[] a, @IndexFor("#1") int i) {
    @Positive
    int length = a.length;
    @Positive
    if (i >= length - 1 || a[i + 1] == null) {
      // body is irrelevant
    @Positive
    }
    @Positive
  }

    @Positive
  public void addition1(String[] a, @IndexFor("#1") int i) {
    @Positive
    int length = a.length;
    @Positive
    if ((i + 1) >= length || a[i + 1] == null) {
      // body is irrelevant
    @Positive
    }
    @Positive
  }

    @Positive
  public void subtraction2(String[] a, @IndexFor("#1") int i) {
    @Positive
    if (i < a.length - 1) {
    @Positive
      @IndexFor("a") int j = i + 1;
    @Positive
    }
    @Positive
  }

    @Positive
  public void addition2(String[] a, @IndexFor("#1") int i) {
    @Positive
    if ((i + 1) < a.length) {
    @Positive
      @IndexFor("a") int j = i + 1;
    @Positive
    }
    @Positive
  }

    @Positive
  public void addition3(String[] a, @IndexFor("#1") int i) {
    @Positive
    if ((i + 5) < a.length) {
    @Positive
      @IndexFor("a") int j = i + 5;
    @Positive
    }
    @Positive
  }

    @Positive
  public void subtraction3(String[] a, @NonNegative int k) {
    @Positive
    if (k - 5 < a.length) {
    @Positive
      String s = a[k - 5];
    @Positive
      @IndexFor("a") int j = k - 5;
    @Positive
    }
    @Positive
  }

    @Positive
  public void subtraction4(String[] a, @IndexFor("#1") int i) {
    @Positive
    if (1 - i < a.length) {
      // The error on this assignment is a false positive.
      // :: error: (assignment)
    @Positive
      @IndexFor("a") int j = 1 - i;

      // :: error: (assignment)
    @Positive
      @LTLengthOf(value = "a", offset = "1") int k = i;
    @Positive
    }
    @Positive
  }

    @Positive
  public void subtraction5(String[] a, int i) {
    @Positive
    if (1 - i < a.length) {
      // :: error: (assignment)
    @Positive
      @IndexFor("a") int j = i;
    @Positive
    }
    @Positive
  }

    @Positive
  public void subtraction6(String[] a, int i, int j) {
    @Positive
    if (i - j < a.length - 1) {
    @Positive
      @IndexFor("a") int k = i - j;
      // :: error: (assignment)
    @Positive
      @IndexFor("a") int k1 = i;
    @Positive
    }
    @Positive
  }

    @Positive
  public void multiplication1(String[] a, int i, @Positive int j) {
    @Positive
    if ((i * j) < (a.length + j)) {
      // :: error: (assignment)
    @Positive
      @IndexFor("a") int k = i;
      // :: error: (assignment)
    @Positive
      @IndexFor("a") int k1 = j;
    @Positive
    }
    @Positive
  }

    @Positive
  public void multiplication2(String @ArrayLen(5) [] a, @IntVal(-2) int i, @IntVal(20) int j) {
    @Positive
    if ((i * j) < (a.length - 20)) {
    @Positive
      @LTLengthOf("a") int k1 = i;
      // :: error: (assignment)
    @Positive
      @LTLengthOf(value = "a", offset = "20") int k2 = i;
      // :: error: (assignment)
    @Positive
      @LTLengthOf("a") int k3 = j;
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
