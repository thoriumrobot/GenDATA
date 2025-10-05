// Test case for https://github.com/typetools/checker-framework/issues/3167 .

    @Positive
public class CharPrintedAsVariable {
    @Positive
  void m1(char c) {
    @Positive
    if (c <= 'A') {
    @Positive
      int x = (int) c;
    @Positive
    }
    @Positive
  }

    @Positive
  void m2(char c) {
    @Positive
    if (c <= '\377') {
    @Positive
      int x = (int) c;
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
