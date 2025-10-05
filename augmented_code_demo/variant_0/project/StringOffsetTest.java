    @Positive
public class StringOffsetTest {
    @Positive
  public static void OffsetString() {
    @Positive
    char[] chars = new char[10];

    // :: error: (argument)
    @Positive
    String string2 = new String(chars, 5, 7);

    @Positive
    String string3 = new String(chars, 5, 4);
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
