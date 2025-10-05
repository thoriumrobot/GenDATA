// Source-based slice around line 109
// Method: <com.google.common.collect.MinMaxPriorityQueue: MinMaxPriorityQueue create()>

 * @since 8.0
 */
@GwtCompatible
public final class MinMaxPriorityQueue<E> extends AbstractQueue<E> {

  /**
   * Creates a new min-max priority queue with default settings: natural order, no maximum size, no
   * initial contents, and an initial expected size of 11.
   */
  public static <E extends Comparable<E>> MinMaxPriorityQueue<E> create() {
    return new Builder<Comparable<E>>(Ordering.natural()).create();
  }

  /**
   * Creates a new min-max priority queue using natural order, no maximum size, and initially
   * containing the given elements.
   */
  public static <E extends Comparable<E>> MinMaxPriorityQueue<E> create(
      Iterable<? extends E> initialContents) {
    return new Builder<E>(Ordering.natural()).create(initialContents);
