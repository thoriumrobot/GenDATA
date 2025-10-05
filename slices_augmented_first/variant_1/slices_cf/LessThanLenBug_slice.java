import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class LessThanLenBug {
    @Positive
  public static void m1(int[] shorter) {
    @Positive
    int[] longer = new int[4 * shorter.length];
    // :: error: (assignment)
    @Positive
    @LTLengthOf("longer") int x = shorter.length;
    @Positive
    int i = longer[x];
    @Positive
  }

    @Positive
  public static void m2(int @MinLen(1) [] shorter) {
    @Positive
    int[] longer = new int[4 * shorter.length];
    @Positive
