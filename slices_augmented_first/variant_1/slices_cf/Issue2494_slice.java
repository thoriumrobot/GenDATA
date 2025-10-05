import org.checkerframework.checker.index.qual.LessThan;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public final class Issue2494 {

    @Positive
  static final long @MinLen(1) [] factorials = {
    @Positive
    1L,
    @Positive
    1L,
    @Positive
    1L * 2,
    @Positive
    1L * 2 * 3,
    @Positive
    1L * 2 * 3 * 4,
