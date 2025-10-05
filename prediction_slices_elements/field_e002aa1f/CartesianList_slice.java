// Source-based slice around line 36
// Method: com.google.common.collect.CartesianList.axes


/**
 * Implementation of {@link Lists#cartesianProduct(List)}.
 *
 * @author Louis Wasserman
 */
@GwtCompatible
final class CartesianList<E> extends AbstractList<List<E>> implements RandomAccess {

  private final transient ImmutableList<List<E>> axes;
  private final transient int[] axesSizeProduct;

  static <E> List<List<E>> create(List<? extends List<? extends E>> lists) {
    ImmutableList.Builder<List<E>> axesBuilder = new ImmutableList.Builder<>(lists.size());
    for (List<? extends E> list : lists) {
      List<E> copy = ImmutableList.copyOf(list);
      if (copy.isEmpty()) {
        return ImmutableList.of();
      }
      axesBuilder.add(copy);
