// Source-based slice around line 37
// Method: com.google.common.collect.RegularImmutableList.EMPTY


/**
 * Implementation of {@link ImmutableList} backed by a simple array.
 *
 * @author Kevin Bourrillion
 */
@GwtCompatible
@SuppressWarnings("serial") // uses writeReplace(), not default serialization
final class RegularImmutableList<E> extends ImmutableList<E> {
  static final ImmutableList<Object> EMPTY = new RegularImmutableList<>(new Object[0]);

  @VisibleForTesting final transient Object[] array;

  RegularImmutableList(Object[] array) {
    this.array = array;
  }

  @Override
  public int size() {
    return array.length;
