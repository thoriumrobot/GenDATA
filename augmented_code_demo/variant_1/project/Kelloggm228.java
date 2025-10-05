    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.LessThan;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class Kelloggm228 {
    @Positive
  public void subList(
    @Positive
      @IndexOrHigh("this") @LessThan("#2 + 1") int fromIndex, @IndexOrHigh("this") int toIndex) {
    @Positive
    if (fromIndex == toIndex) {
    @Positive
      return;
    @Positive
    }

    @Positive
    @Positive int x = toIndex;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
