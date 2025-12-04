    @GTENegativeOne
    @NonNegative
    @Positive
    @GTENegativeOne
    @NonNegative
    @Positive
    @GTENegativeOne
    @NonNegative
    @Positive
    @GTENegativeOne
    @NonNegative
    @Positive
    @NonNegative
    @NonNegative
    @NonNegative
    @NonNegative
    @GTENegativeOne
    @NonNegative
    @Positive
    @GTENegativeOne
    @NonNegative
    @Positive
    @GTENegativeOne
    @NonNegative
    @Positive
    @GTENegativeOne
    @NonNegative
    @Positive
    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;

    @Positive
public class AndExample {

    @Positive
  private static final @IndexOrHigh("iYearInfoCache") int CACHE_SIZE = 1 << 10;

    @Positive
  private static final @IndexFor("iYearInfoCache") int CACHE_MASK = CACHE_SIZE - 1;

    @Positive
  private static final String[] iYearInfoCache = new String[CACHE_SIZE];

    @Positive
  private String getYearInfo(int year) {
    @Positive
    return iYearInfoCache[year & CACHE_MASK];
    @Positive
  }
    @Positive
}
