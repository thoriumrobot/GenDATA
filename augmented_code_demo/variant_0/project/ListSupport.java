    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.UpperBoundBottom;

// @skip-test until we bring list support back

    @Positive
public class ListSupport {

    @Positive
  void indexOf(List<Integer> list) {
    @Positive
    int index = list.indexOf(0);

    @Positive
    @LTLengthOf("list") int i = index;

    // :: error: (assignment)
    @Positive
    @UpperBoundBottom int i2 = index;
    @Positive
  }

    @Positive
  void lastIndexOf(List<Integer> list) {
    @Positive
    int index = list.lastIndexOf(0);

    @Positive
    @LTLengthOf("list") int i = index;

    // :: error: (assignment)
    @Positive
    @UpperBoundBottom int i2 = index;
    @Positive
  }

    @Positive
  void subList(List<Integer> list, @LTLengthOf("#1") int index, @LTEqLengthOf("#1") int endIndex) {
    @Positive
    List<Integer> list2 = list.subList(index, endIndex);

    // start index must be strictly lessthanlength
    // :: error: (argument)
    @Positive
    list2 = list.subList(endIndex, endIndex);

    // edindex must be less than or equal to Length
    // :: error: (argument)
    @Positive
    list2 = list.subList(index, 28);
    @Positive
  }

    @Positive
  void size(List<Integer> list) {
    @Positive
    int i = list.size();
    @Positive
    @LTEqLengthOf("list") int k = i;

    // :: error: (assignment)
    @Positive
    @LTLengthOf("list") int m = i;
    @Positive
  }

    @Positive
  void clear(List<Integer> list) {
    @Positive
    int lessThanLength = list.size() - 1;
    @Positive
    int lessThanOrEq = list.size();
    @Positive
    list.get(lessThanLength);

    @Positive
    list.clear();

    // :: error: (list.access.unsafe.high)
    @Positive
    list.get(lessThanLength);

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("list") int m = lessThanLength;

    // :: error: (assignment)
    @Positive
    m = lessThanOrEq;

    // :: error: (assignment)
    @Positive
    @LTLengthOf("list") int i = lessThanLength;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
