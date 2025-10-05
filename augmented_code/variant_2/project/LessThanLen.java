/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.*;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class LessThanLen {

    @Positive
  public static void m1() {
    @Positive
    int[] shorter = new int[5];
    @Positive
    int[] longer = new int[shorter.length * 2];
    @Positive
    int i = 0;
        while (i < shorter.length) {
            @Positive
      longer[i] = shorter[i];
    @Positive
            i++;
        }
    @Positive
  }

    @Positive
  public static void m2(int @MinLen(1) [] shorter) {
    @Positive
    int[] longer = new int[shorter.length * 2];
    @Positive
    int i = 0;
        while (i < shorter.length) {
            @Positive
      longer[i] = shorter[i];
    @Positive
            i++;
        }
    @Positive
  }

    @Positive
  public static void m3(int[] shorter) {
    @Positive
    int[] longer = new int[shorter.length + 1];
    @Positive
    int i = 0;
        while (i < shorter.length) {
            @Positive
      longer[i] = shorter[i];
    @Positive
            i++;
        }
    @Positive
  }

    @Positive
  public static void m4(int @MinLen(1) [] shorter) {
    @Positive
    int[] longer = new int[shorter.length * 1];
    // :: error: (assignment)
    @Positive
    @LTLengthOf("longer") int x = shorter.length;
    @Positive
    @LTEqLengthOf("longer") int y = shorter.length;
    @Positive
  }

    @Positive
  public static void m5(int[] shorter) {
    // :: error: (array.length.negative)
    @Positive
    int[] longer = new int[shorter.length * -1];
    // :: error: (assignment)
    @Positive
    @LTLengthOf("longer") int x = shorter.length;
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("longer") int y = shorter.length;
    @Positive
  }

    @Positive
  public static void m6(int @MinLen(1) [] shorter) {
    @Positive
    int[] longer = new int[4 * shorter.length];
    // TODO: enable when https://github.com/kelloggm/checker-framework/issues/211 is fixed
    // // :: error: (assignment)
    // @LTLengthOf("longer") int x = shorter.length;
    @Positive
    @LTEqLengthOf("longer") int y = shorter.length;
    @Positive
  }

    @Positive
  public static void m7(int @MinLen(1) [] shorter) {
    @Positive
    int[] longer = new int[4 * shorter.length];
    @Positive
    @LTLengthOf("longer") int x = shorter.length;
    @Positive
    @LTEqLengthOf("longer") int y = shorter.length;
    @Positive
  }
    @Positive
}
