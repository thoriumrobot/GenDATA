// Source-based slice around line 31
// Method: com.google.common.collect.EmptyImmutableSetMultimap.INSTANCE

import java.util.Collection;

/**
 * Implementation of {@link ImmutableListMultimap} with no entries.
 *
 * @author Mike Ward
 */
@GwtCompatible
final class EmptyImmutableSetMultimap extends ImmutableSetMultimap<Object, Object> {
  static final EmptyImmutableSetMultimap INSTANCE = new EmptyImmutableSetMultimap();

  private EmptyImmutableSetMultimap() {
    super(ImmutableMap.of(), 0, null);
  }

  /*
   * TODO(b/242884182): Figure out why this helps produce the same class file when we compile most
   * of common.collect a second time with the results of the first compilation on the classpath. Or
   * just back this out once we stop doing that (which we'll do after our internal GWT setup
   * changes).
