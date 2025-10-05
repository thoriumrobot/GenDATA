// Source-based slice around line 42
// Method: <com.google.common.collect.ImmutableEnumSet: ImmutableSet asImmutable(EnumSet)>


/**
 * Implementation of {@link ImmutableSet} backed by a non-empty {@link java.util.EnumSet}.
 *
 * @author Jared Levy
 */
@GwtCompatible
@SuppressWarnings("serial") // we're overriding default serialization
final class ImmutableEnumSet<E extends Enum<E>> extends ImmutableSet<E> {
  static <E extends Enum<E>> ImmutableSet<E> asImmutable(EnumSet<E> set) {
    switch (set.size()) {
      case 0:
        return ImmutableSet.of();
      case 1:
        return ImmutableSet.of(getOnlyElement(set));
      default:
        return new ImmutableEnumSet<>(set);
    }
  }

