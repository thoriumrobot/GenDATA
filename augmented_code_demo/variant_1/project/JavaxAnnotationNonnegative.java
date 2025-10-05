// Test for https://github.com/typetools/checker-framework/issues/6507

    @Positive
public class JavaxAnnotationNonnegative {

    @Positive
  public static void test(@javax.annotation.Nonnegative int y) {
    @Positive
    get(y);
    @Positive
  }

    @Positive
  public static void get(@org.checkerframework.checker.index.qual.NonNegative int x) {}
    @Positive
}

// CFWR semantic augmentation - variant 1
