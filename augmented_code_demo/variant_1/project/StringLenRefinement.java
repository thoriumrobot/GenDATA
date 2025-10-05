    @Positive
import org.checkerframework.common.value.qual.ArrayLen;
    @Positive
import org.checkerframework.common.value.qual.ArrayLenRange;
    @Positive
import org.checkerframework.common.value.qual.StringVal;

    @Positive
public class StringLenRefinement {

    @Positive
  void refineLenRange(
    @Positive
      @ArrayLenRange(from = 3, to = 10) String range,
    @Positive
      @ArrayLen({4, 6, 12}) String lens,
    @Positive
      @StringVal({"aaaa", "bbbb", "cccccc", "dddddddddddd"}) String vals) {
    @Positive
    if (range.length() <= 7) {
    @Positive
      @ArrayLenRange(from = 3, to = 7) String shortRange = range;
    @Positive
    } else {
    @Positive
      @ArrayLenRange(from = 8, to = 10) String longRange = range;
    @Positive
    }

    @Positive
    if (lens.length() <= 7) {
    @Positive
      @ArrayLen({4, 6}) String shortLens = lens;
    @Positive
    } else {
    @Positive
      @ArrayLen({12}) String longLens = lens;
    @Positive
    }

    @Positive
    if (vals.length() <= 7) {
    @Positive
      @StringVal({"aaaa", "bbbb", "cccccc"}) String shortVals = vals;
    @Positive
    } else {

    @Positive
      @StringVal({"dddddddddddd"}) String longVals = vals;
    @Positive
    }
    @Positive
  }

    @Positive
  void refineLen(
    @Positive
      @ArrayLenRange(from = 3, to = 10) String range, @ArrayLen({4, 8, 12}) String lens) {

    @Positive
    if (range.length() == 5 || range.length() == 8 || range.length() == 13) {
    @Positive
      @ArrayLen({5, 8}) String refinedArg = range;
    @Positive
    }

    @Positive
    if (lens.length() == 5 || lens.length() == 8 || lens.length() == 13) {
    @Positive
      @ArrayLen({8}) String refinedLens = lens;
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
