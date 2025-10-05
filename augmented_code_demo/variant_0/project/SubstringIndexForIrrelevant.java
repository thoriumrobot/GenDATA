    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.SubstringIndexFor;

    @Positive
public class SubstringIndexForIrrelevant {

    @Positive
      "substringindex:return" // https://github.com/kelloggm/checker-framework/issues/206,
  // 207, 208
    @Positive
  )
    @Positive
  public static @LTEqLengthOf("#1") @SubstringIndexFor(value = "#1", offset = "#2.length - 1") int
    @Positive
      indexOf(boolean[] array, boolean[] target) {
    @Positive
    return -1;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
