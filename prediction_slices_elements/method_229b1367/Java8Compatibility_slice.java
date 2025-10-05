// Source-based slice around line 28
// Method: <com.google.common.io.Java8Compatibility: void clear(Buffer)>

import java.nio.Buffer;

/**
 * Wrappers around {@link Buffer} methods that are covariantly overridden in Java 9+. See
 * https://github.com/google/guava/issues/3990
 */
@J2ktIncompatible
@GwtIncompatible
final class Java8Compatibility {
  static void clear(Buffer b) {
    b.clear();
  }

  static void flip(Buffer b) {
    b.flip();
  }

  static void limit(Buffer b, int limit) {
    b.limit(limit);
  }
