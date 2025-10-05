    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LessThan;

    @Positive
public class Issue2613 {

    @Positive
  private static final String STRING_CONSTANT = "Hello";

    @Positive
  void integerConstant() {
    @Positive
    require_lt(0, Integer.MAX_VALUE);
    @Positive
  }

    @Positive
  void StringConstant() {
    @Positive
    require_lt(0, STRING_CONSTANT);
    @Positive
  }

    @Positive
  void require_lt(@LessThan("#2") int a, int b) {}

    @Positive
  void require_lt(@LTLengthOf("#2") int a, String b) {}

    @Positive
  void method(@LessThan("Integer.MAX_VALUE") long x, @LessThan("Integer.MAX_VALUE") long y) {
    @Positive
    x = y;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
