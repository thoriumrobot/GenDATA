/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import java.util.Formattable;
    @Positive
import java.util.LinkedHashMap;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.stream.Collector;
    @Positive
import java.util.stream.Collectors;
    @Positive
import org.checkerframework.common.value.qual.ArrayLen;

    @Positive
public class IndexIssue6046 {

    @Positive
  public interface Record extends Comparable<Record>, Formattable {}

    @Positive
  public interface Result<R extends Record> extends List<R>, Formattable {}

    @Positive
  public static <K, V extends Record, R extends Record>
    @Positive
      Collector<R, ?, Map<K, Result<V>>> intoResultGroups(
    @Positive
          Function<? super R, ? extends K> keyMapper) {

    @Positive
    return Collectors.groupingBy(
    @Positive
        keyMapper,
    @Positive
        LinkedHashMap::new,
    @Positive
        Collector.<R, Result<V>[], Result<V>>of(
            // :: error:  (array.access.unsafe.high.constant)
    @Positive
            () -> new Result[1], (x, r) -> {}, (r1, r2) -> r1, r -> r[0]));
    @Positive
  }

    @Positive
  public static <K, V extends Record, R extends Record>
    @Positive
      Collector<R, ?, Map<K, Result<V>>> intoResultGroups2(
    @Positive
          Function<? super R, ? extends K> keyMapper) {

    @Positive
    return Collectors.groupingBy(
    @Positive
        keyMapper,
    @Positive
        LinkedHashMap::new,
    @Positive
        Collector.<R, Result<V> @ArrayLen(1) [], Result<V>>of(
    @Positive
            () -> new Result[1], (x, r) -> {}, (r1, r2) -> r1, r -> r[0]));
    @Positive
  }

    @Positive
  public static <R extends Record> Result<R> result(R record) {
    @Positive
    throw new RuntimeException();
    @Positive
  }
    @Positive
}
