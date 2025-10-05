// Source-based slice around line 54
// Method: <com.google.common.collect.EnumMultiset: EnumMultiset create(Class)>

 * @author Jared Levy
 * @since 2.0
 */
@GwtCompatible
@J2ktIncompatible
@SuppressWarnings("EnumOrdinal") // This is one of the low-level utilities where it's suitable.
public final class EnumMultiset<E extends Enum<E>> extends AbstractMultiset<E>
    implements Serializable {
  /** Creates an empty {@code EnumMultiset}. */
  public static <E extends Enum<E>> EnumMultiset<E> create(Class<E> type) {
    return new EnumMultiset<>(type);
  }

  /**
   * Creates a new {@code EnumMultiset} containing the specified elements.
   *
   * <p>This implementation is highly efficient when {@code elements} is itself a {@link Multiset}.
   *
   * @param elements the elements that the multiset should contain
   * @throws IllegalArgumentException if {@code elements} is empty
