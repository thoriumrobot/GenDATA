// Source-based slice around line 44
// Method: com.google.common.collect.AbstractSequentialIterator.nextOrNull

 *       }
 *     };
 * }
 *
 * @author Chris Povirk
 * @since 12.0 (in Guava as {@code AbstractLinkedIterator} since 8.0)
 */
@GwtCompatible
public abstract class AbstractSequentialIterator<T> extends UnmodifiableIterator<T> {
  private @Nullable T nextOrNull;

  /**
   * Creates a new iterator with the given first element, or, if {@code firstOrNull} is null,
   * creates a new empty iterator.
   */
  protected AbstractSequentialIterator(@Nullable T firstOrNull) {
    this.nextOrNull = firstOrNull;
  }

  /**
