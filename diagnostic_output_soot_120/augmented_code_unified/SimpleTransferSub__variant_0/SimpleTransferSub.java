    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class SimpleTransferSub {
    @Positive
  void test() {
    // shows a bug in the Checker Framework. I don't think we can get around this bit...
    @Positive
    int bs = 0;
    // :: error: (assignment)
    @Positive
    @Positive int ds = bs--;
    @Positive
  }
    @Positive
}
// a comment
