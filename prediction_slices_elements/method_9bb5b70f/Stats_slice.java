// Source-based slice around line 460
// Method: <com.google.common.math.Stats: double sumOfSquaresOfDeltas()>

          .add("populationStandardDeviation", populationStandardDeviation())
          .add("min", min)
          .add("max", max)
          .toString();
    } else {
      return MoreObjects.toStringHelper(this).add("count", count).toString();
    }
  }

  double sumOfSquaresOfDeltas() {
    return sumOfSquaresOfDeltas;
  }

  /**
   * Returns the <a href="http://en.wikipedia.org/wiki/Arithmetic_mean">arithmetic mean</a> of the
   * values. The count must be non-zero.
   *
   * <p>The definition of the mean is the same as {@link Stats#mean}.
   *
   * @param values a series of values, which will be converted to {@code double} values (this may
