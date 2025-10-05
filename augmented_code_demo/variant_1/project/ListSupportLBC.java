    @Positive
import java.util.ArrayList;
    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

// @skip-test until we bring list support back

    @Positive
public class ListSupportLBC {

    @Positive
  void testGet() {

    @Positive
    List<Integer> list = new ArrayList<>();
    @Positive
    int i = -1;
    @Positive
    int j = 0;

    // try and use a negative to get, should fail
    // :: error: (argument)
    @Positive
    Integer m = list.get(i);

    // try and use a nonnegative, should work
    @Positive
    Integer l = list.get(j);
    @Positive
  }

    @Positive
  void testArrayListGet() {

    @Positive
    ArrayList<Integer> list = new ArrayList<>();
    @Positive
    int i = -1;
    @Positive
    int j = 0;

    // try and use a negative to get, should fail
    // :: error: (argument)
    @Positive
    Integer m = list.get(i);

    // try and use a nonnegative, should work
    @Positive
    Integer l = list.get(j);
    @Positive
  }

    @Positive
  void testSet() {
    @Positive
    List<Integer> list = new ArrayList<>();
    @Positive
    int i = -1;
    @Positive
    int j = 0;

    // try and use a negative to get, should fail
    // :: error: (argument)
    @Positive
    Integer m = list.set(i, 34);

    // try and use a nonnegative, should work
    @Positive
    Integer l = list.set(j, 34);
    @Positive
  }

    @Positive
  void testIndexOf() {
    @Positive
    List<Integer> list = new ArrayList<>();
    @Positive
    @GTENegativeOne int a = list.indexOf(1);

    // :: error: (assignment)
    @Positive
    @NonNegative int n = a;

    @Positive
    @GTENegativeOne int b = list.lastIndexOf(1);

    // :: error: (assignment)
    @Positive
    @NonNegative int m = b;
    @Positive
  }

    @Positive
  void testSize() {
    @Positive
    List<Integer> list = new ArrayList<>();
    @Positive
    @NonNegative int s = list.size();

    // :: error: (assignment)
    @Positive
    @Positive int r = s;
    @Positive
  }

    @Positive
  void testSublist() {
    @Positive
    List<Integer> list = new ArrayList<>();
    @Positive
    int i = -1;
    @Positive
    int j = 0;

    // :: error: (argument)
    @Positive
    List<Integer> k = list.subList(i, i);

    // :: error: (argument)
    @Positive
    List<Integer> a = list.subList(i, j);

    // :: error: (argument)
    @Positive
    List<Integer> b = list.subList(j, i);

    // should work since both are nonnegative
    @Positive
    List<Integer> c = list.subList(j, j);
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
