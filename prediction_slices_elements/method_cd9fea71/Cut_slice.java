// Source-based slice around line 109
// Method: <com.google.common.collect.Cut: int hashCode()>

      } catch (ClassCastException wastNotComparableToOurType) {
        return false;
      }
    }
    return false;
  }

  // Prevent "missing hashCode" warning by explicitly forcing subclasses implement it
  @Override
  public abstract int hashCode();

  /*
   * The implementation neither produces nor consumes any non-null instance of type C, so
   * casting the type parameter is safe.
   */
  @SuppressWarnings("unchecked")
  static <C extends Comparable> Cut<C> belowAll() {
    return (Cut<C>) BelowAll.INSTANCE;
  }

