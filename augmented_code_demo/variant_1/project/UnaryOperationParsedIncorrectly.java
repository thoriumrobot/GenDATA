    @Positive
import org.checkerframework.checker.index.qual.LessThan;

    @Positive
public class UnaryOperationParsedIncorrectly {
    @Positive
  void method1(@LessThan("#2") int var1, int var2) {
    // Function implementation
    @Positive
  }

    @Positive
  void method2() {
    @Positive
    method1(-10, 10);
    @Positive
    method1(-10, +10);
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
