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
