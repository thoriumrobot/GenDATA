import java.lang.reflect.Array;
import org.checkerframework.common.value.qual.MinLen;

public class ReflectArray {

    void testMinLen(Object @MinLen(1) [] a) {
        Array.get(a, 0);
        Array.get(a, 1);
    }
}
