// Tests that String.length() is supported in the same situations as array length

    @Positive
import java.util.Random;
    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;
    @Positive
import org.checkerframework.checker.index.qual.SameLen;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class StringLength {
    @Positive
  void testMinLenSubtractPositive(@MinLen(10) String s) {
    @Positive
    @Positive int i1 = s.length() - 9;
    @Positive
    @NonNegative int i0 = s.length() - 10;
    // ::  error: (assignment)
    @Positive
    @NonNegative int im1 = s.length() - 11;
    @Positive
  }

    @Positive
  void testNewArraySameLen(String s) {
    @Positive
    int @SameLen("s") [] array = new int[s.length()];
    // ::  error: (assignment)
    @Positive
    int @SameLen("s") [] array1 = new int[s.length() + 1];
    @Positive
  }

    @Positive
  void testStringAssignSameLen(String s, String r) {
    @Positive
    @SameLen("s") String t = s;
    // ::  error: (assignment)
    @Positive
    @SameLen("s") String tN = r;
    @Positive
  }

    @Positive
  void testStringLenEqualSameLen(String s, String r) {
    @Positive
    if (s.length() == r.length()) {
    @Positive
      @SameLen("s") String tN = r;
    @Positive
    }
    @Positive
  }

    @Positive
  void testStringEqualSameLen(String s, String r) {
    @Positive
    if (s == r) {
    @Positive
      @SameLen("s") String tN = r;
    @Positive
    }
    @Positive
  }

    @Positive
  void testOffsetRemoval(
    @Positive
      String s,
    @Positive
      String t,
    @Positive
      @LTLengthOf(value = "#1", offset = "#2.length()") int i,
    @Positive
      @LTLengthOf(value = "#2") int j,
    @Positive
      int k) {
    @Positive
    @LTLengthOf("s") int ij = i + j;
    // ::  error: (assignment)
    @Positive
    @LTLengthOf("s") int ik = i + k;
    @Positive
  }

    @Positive
  void testLengthDivide(@MinLen(1) String s) {
    @Positive
    @IndexFor("s") int i = s.length() / 2;
    @Positive
  }

    @Positive
  void testAddDivide(@MinLen(1) String s, @IndexFor("#1") int i, @IndexFor("#1") int j) {
    @Positive
    @IndexFor("s") int ij = (i + j) / 2;
    @Positive
  }

    @Positive
  void testRandomMultiply(@MinLen(1) String s, Random r) {
    @Positive
    @LTLengthOf("s") int i = (int) (Math.random() * s.length());
    @Positive
    @LTLengthOf("s") int j = (int) (r.nextDouble() * s.length());
    @Positive
  }

    @Positive
  void testNotEqualLength(String s, @IndexOrHigh("#1") int i, @IndexOrHigh("#1") int j) {
    @Positive
    if (i != s.length()) {
    @Positive
      @IndexFor("s") int in = i;
      // ::  error: (assignment)
    @Positive
      @IndexFor("s") int jn = j;
    @Positive
    }
    @Positive
  }

    @Positive
  void testLength(String s) {
    @Positive
    @IndexOrHigh("s") int i = s.length();
    @Positive
    @LTLengthOf("s") int j = s.length() - 1;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
