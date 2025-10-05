// Source-based slice around line 329
// Method: <com.google.common.util.concurrent.CycleDetectingLockFactory: String getLockName(Enum)>

      nodes.get(i).checkAcquiredLocks(Policies.DISABLED, nodes.subList(i + 1, numKeys));
    }
    return Collections.unmodifiableMap(map);
  }

  /**
   * For the given Enum value {@code rank}, returns the value's {@code "EnumClass.name"}, which is
   * used in exception and warning output.
   */
  private static String getLockName(Enum<?> rank) {
    return rank.getDeclaringClass().getSimpleName() + "." + rank.name();
  }

  /**
   * A {@code CycleDetectingLockFactory.WithExplicitOrdering} provides the additional enforcement of
   * an application-specified ordering of lock acquisitions. The application defines the allowed
   * ordering with an {@code Enum} whose values each correspond to a lock type. The order in which
   * the values are declared dictates the allowed order of lock acquisition. In other words, locks
   * corresponding to smaller values of {@link Enum#ordinal()} should only be acquired before locks
   * with larger ordinals. Example:
