// Source-based slice around line 462
// Method: <com.google.common.math.Quantiles: boolean containsNaN(double)>

          ret.put(
              indexes[i], interpolate(dataset[quotient], dataset[quotient + 1], remainder, scale));
        }
      }
      return unmodifiableMap(ret);
    }
  }

  /** Returns whether any of the values in {@code dataset} are {@code NaN}. */
  private static boolean containsNaN(double... dataset) {
    for (double value : dataset) {
      if (Double.isNaN(value)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Returns a value a fraction {@code (remainder / scale)} of the way between {@code lower} and
