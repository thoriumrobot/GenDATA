/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import java.util.Arrays;
    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class BinarySearchTest {

    @Positive
  private final long @SameLen("iNameKeys") [] iTransitions;
    @Positive
  private final String @SameLen("iTransitions") [] iNameKeys;

    @Positive
  private BinarySearchTest(
    @Positive
      long @SameLen("iNameKeys") [] transitions, String @SameLen("iTransitions") [] nameKeys) {
    @Positive
    iTransitions = transitions;
    @Positive
    iNameKeys = nameKeys;
    @Positive
  }

    @Positive
  public String getNameKey(long instant) {
    @Positive
    long[] transitions = iTransitions;
    @Positive
    int i = Arrays.binarySearch(transitions, instant);
    @Positive
    if (i >= 0) {
    @Positive
      return iNameKeys[i];
    @Positive
    }
    @Positive
    i = ~i;
    @Positive
    if (i > 0) {
    @Positive
      return iNameKeys[i - 1];
    @Positive
    }
    @Positive
    return "";
    @Positive
  }

    @Positive
  public String getNameKey2(long instant) {
    @Positive
    long[] transitions = iTransitions;
    @Positive
    int i = Arrays.binarySearch(transitions, instant);
    @Positive
    if (i >= 0) {
    @Positive
      return iNameKeys[i];
    @Positive
    }
    @Positive
    i = ~i;
    @Positive
    if (i < iNameKeys.length) {
    @Positive
      return iNameKeys[i];
    @Positive
    }
    @Positive
    return "";
    @Positive
  }
    @Positive
}
