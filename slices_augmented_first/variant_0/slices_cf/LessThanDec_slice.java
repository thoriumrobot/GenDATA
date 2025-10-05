    @Positive
import org.checkerframework.checker.index.qual.IndexOrLow;
    @Positive
import org.checkerframework.checker.index.qual.LessThan;

    @Positive
public class LessThanDec {
    @Positive
  private static @IndexOrLow("#1") @LessThan("#4") int lastIndexOf(
    @Positive
      short[] array, short target, @IndexOrHigh("#1") int start, @IndexOrHigh("#1") int end) {
    @Positive
    int i = end - 1;
        while (i >= start) {
            @Positive
      if (array[i] == target) {
    @Positive
        return i;
    @Positive
      }
    @Positive
