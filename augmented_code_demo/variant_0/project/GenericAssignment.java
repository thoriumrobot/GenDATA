    @Positive
import java.util.List;
    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;
    @Positive
import org.checkerframework.common.value.qual.IntRange;

    @Positive
public class GenericAssignment {
    @Positive
  public void assignNonNegativeList(List<@NonNegative Integer> l) {
    @Positive
    List<@NonNegative Integer> i = l; // line 10
    @Positive
  }

    @Positive
  public void assignPositiveList(List<@Positive Integer> l) {
    @Positive
    List<@Positive Integer> i = l; // line 13
    @Positive
  }

    @Positive
  public void assignGTENOList(List<@GTENegativeOne Integer> l) {
    @Positive
    List<@GTENegativeOne Integer> i = l; // line 16
    @Positive
  }

  // Similar examples that work
    @Positive
  public void assignNonNegativeArrayOK(@NonNegative Integer[] l) {
    @Positive
  }

    @Positive
  public void assignIntRangeListOK(List<@IntRange(from = 0) Integer> l) {
    @Positive
    List<@IntRange(from = 0) Integer> i = l;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
