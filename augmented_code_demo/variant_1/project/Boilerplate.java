    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class Boilerplate {

    @Positive
  void test() {
    // :: error: (assignment)
    @Positive
    @Positive int a = -1;
    @Positive
  }
    @Positive
}
// a comment

// CFWR semantic augmentation - variant 1
