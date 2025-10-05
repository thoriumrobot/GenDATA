    @Positive
public class Dimension {
    @Positive
  void test(int expr) {
    @Positive
    int[] array = new int[expr];
    // :: error: (array.access.unsafe.high)
    @Positive
    array[expr] = 0;
    @Positive
    array[expr - 1] = 0;
    @Positive
  }

    @Positive
  String[] arrayField = new String[1];

    @Positive
  void test2(int expr) {
    @Positive
    arrayField = new String[expr];
    // :: error: (array.access.unsafe.high)
    @Positive
    this.arrayField[expr] = "";
    @Positive
    this.arrayField[expr - 1] = "";
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
