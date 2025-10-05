/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// Test case for https://github.com/typetools/checker-framework/issues/5471.

    @Positive
import org.checkerframework.checker.index.qual.IndexFor;

    @Positive
public class Issue5471 {
    @Positive
  private static boolean atTheBeginning(@IndexFor("#2") int index, String line) {
    @Positive
    return (index == 0);
    @Positive
  }

    @Positive
  private static boolean hasDoubleQuestionMarkAtTheBeginning(String line) {
    @Positive
    int i = line.indexOf("??");
    @Positive
    if (i != -1) {
    @Positive
      return (atTheBeginning(i, line));
    @Positive
    }
    @Positive
    return false;
    @Positive
  }

    @Positive
  public static void main(String[] args) {
    @Positive
    String x = "Hello?World, this is our new program";
    @Positive
    if (hasDoubleQuestionMarkAtTheBeginning(x)) System.out.println("TRUE");
    @Positive
  }
    @Positive
}
