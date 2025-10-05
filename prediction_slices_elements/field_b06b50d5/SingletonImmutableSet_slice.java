// Source-based slice around line 39
// Method: com.google.common.collect.SingletonImmutableSet.element

 * @author Kevin Bourrillion
 * @author Nick Kralevich
 */
@GwtCompatible
@SuppressWarnings("serial") // uses writeReplace(), not default serialization
final class SingletonImmutableSet<E> extends ImmutableSet<E> {
  // We deliberately avoid caching the asList and hashCode here, to ensure that with
  // compressed oops, a SingletonImmutableSet packs all the way down to the optimal 16 bytes.

  final transient E element;

  SingletonImmutableSet(E element) {
    this.element = Preconditions.checkNotNull(element);
  }

  @Override
  public int size() {
    return 1;
  }

