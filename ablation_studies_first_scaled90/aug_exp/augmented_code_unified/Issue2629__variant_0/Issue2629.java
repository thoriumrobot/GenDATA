// Test case for Issue 2629
// https://github.com/typetools/checker-framework/issues/2629

    @Positive
import org.checkerframework.checker.index.qual.LessThan;

    @Positive
public class Issue2629 {
    @Positive
  @LessThan("#1 + 1") int test(int a) {
    @Positive
    return a;
    @Positive
  }
    @Positive
}
