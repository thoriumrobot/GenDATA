/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
